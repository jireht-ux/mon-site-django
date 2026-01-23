import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json
import re
import os

# 1. Configuration et Chargement des modèles
# apply_ocr=False car nous utilisons PyMuPDF pour extraire le texte proprement
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_invoice(pdf_path):
    """Analyse le PDF et extrait des blocs structurés par clustering spatial strict."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Le fichier {pdf_path} est introuvable.")

    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        raise ValueError("Le document PDF est vide ou corrompu.")
    
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    raw_words_data = page.get_text("words")
    if not raw_words_data: return []
    
    # Tri spatial initial : haut en bas, puis gauche à droite
    words_data = sorted(raw_words_data, key=lambda w: (w[1], w[0]))
    
    words = [w[4] for w in words_data]
    raw_boxes = [[w[0], w[1], w[2], w[3]] for w in words_data]
    
    # NORMALISATION SÉCURISÉE (Fix Erreur 0-1000)
    w_scale, h_scale = 1000 / page_w, 1000 / page_h
    normalized_boxes = []
    for b in raw_boxes:
        normalized_boxes.append([
            max(0, min(1000, int(b[0] * w_scale))),
            max(0, min(1000, int(b[1] * h_scale))),
            max(0, min(1000, int(b[2] * w_scale))),
            max(0, min(1000, int(b[3] * h_scale)))
        ])

    # Préparation pour le modèle
    encoding = processor(img, words, boxes=normalized_boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    labels = [model.config.id2label[p] for p in predictions[:len(words)]]

    tokens_info = [{'text': t, 'label': l, 'bbox': b} for t, l, b in zip(words, labels, raw_boxes)]
    
    # CLUSTERING DE BASE (Formation des lignes et petits paragraphes)
    def get_clusters(data, threshold_x=30, threshold_y=10):
        clusters = []
        if not data: return clusters
        
        # On s'assure que le tri est respecté pour le clustering
        data = sorted(data, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        curr_cluster = [data[0]]
        for i in range(1, len(data)):
            prev, curr = data[i-1]['bbox'], data[i]['bbox']
            # Si même ligne et proche horizontalement
            if abs(curr[1] - prev[1]) < threshold_y and (curr[0] - prev[2]) < threshold_x:
                curr_cluster.append(data[i])
            else:
                clusters.append(curr_cluster)
                curr_cluster = [data[i]]
        clusters.append(curr_cluster)
        return clusters

    raw_clusters = get_clusters(tokens_info)
    structured_blocks = []

    for cluster in raw_clusters:
        full_text = " ".join([w['text'] for w in cluster])
        clean_labels = [w['label'].replace("B-","").replace("I-","") for w in cluster if w['label'] != "O"]
        main_label = max(set(clean_labels), key=clean_labels.count) if clean_labels else "O"
        
        structured_blocks.append({
            "label_IA": main_label,
            "top": round(min(w['bbox'][1] for w in cluster) / page_h, 4),
            "bottom": round(max(w['bbox'][3] for w in cluster) / page_h, 4),
            "left": round(min(w['bbox'][0] for w in cluster) / page_w, 4),
            "right": round(max(w['bbox'][2] for w in cluster) / page_w, 4),
            "content": full_text
        })

    return structured_blocks

def generate_dcp_report(blocks):
    """Liaison intelligente Anchor-Value sans effet cascade."""
    if not blocks: return {}

    merged_results = []
    skip = set()
    anchor_keywords = ["TOTAL", "TTC", "HT", "TVA", "N°", "DATE", "ÉCHÉANCE", "MONTANT", "FACTURE"]
    
    sorted_blocks = sorted(blocks, key=lambda b: (b['top'], b['left']))

    for i in range(len(sorted_blocks)):
        if i in skip: continue
        # On utilise une copie pour éviter de modifier la liste en cours de lecture
        curr = dict(sorted_blocks[i])
        content_up = curr['content'].upper()
        
        # Si le bloc contient un mot-clé, on cherche SA valeur unique
        if any(key in content_up for key in anchor_keywords):
            target_idx = -1
            
            # 1. Recherche Horizontale (Même ligne, à droite)
            for j in range(i + 1, len(sorted_blocks)):
                if j in skip: continue
                cand = sorted_blocks[j]
                if abs(curr['top'] - cand['top']) < 0.012 and cand['left'] > curr['left']:
                    target_idx = j
                    break # On prend le premier et on s'arrête
            
            # 2. Recherche Verticale (Juste en dessous, aligné)
            if target_idx == -1:
                for j in range(i + 1, len(sorted_blocks)):
                    if j in skip: continue
                    cand = sorted_blocks[j]
                    # Espace vertical max 3% et alignement des bords gauches
                    if 0 <= (cand['top'] - curr['bottom']) < 0.035 and abs(curr['left'] - cand['left']) < 0.10:
                        target_idx = j
                        break
            
            # Fusionner si une cible a été trouvée
            if target_idx != -1:
                target_block = sorted_blocks[target_idx]
                curr['content'] += f" : {target_block['content']}"
                curr['right'] = max(curr['right'], target_block['right'])
                curr['bottom'] = max(curr['bottom'], target_block['bottom'])
                skip.add(target_idx)

        merged_results.append(curr)

    # 2. Audit Final et Filtrage des Données Pertinentes
    dcp_report = {}
    patterns = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "PHONE": r'(?:\+237|237)?\s*[2368]\s*[0-9](?:\s*[0-9]{2}){3}',
        "SIRET": r'\d{14}',
        "TVA": r'[A-Z]{2}\d{11}'
    }

    for i, block in enumerate(merged_results):
        content = block['content']
        final_label = block['label_IA']
        is_sensitive = False

        # Vérification par Regex
        for d_name, p in patterns.items():
            if re.search(p, content):
                final_label = f"ZONE_{d_name}"
                is_sensitive = True
                break
        
        # Qualification sémantique
        up_content = content.upper()
        if any(kw in up_content for kw in ["SARL", "SAS", "ETABLISSEMENT"]):
            final_label = "SOCIETE_EMETTEUR"
        
        if "CLIENT" in up_content or is_sensitive:
            is_sensitive = True

        # On n'exporte que ce qui est utile (Pas de "O" ou d'infos inutiles)
        if final_label != "O" or is_sensitive:
            key = f"{final_label}_{i}"
            dcp_report[key] = {
                "top": block['top'],
                "bottom": block['bottom'],
                "left": block['left'],
                "right": block['right'],
                "content": content,
                "is_sensitive": is_sensitive
            }

    return dcp_report

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        # TEST : vérifiez bien le nom du fichier
        file_path = "facture_23.pdf"
        
        print(f"--- ANALYSE DE : {file_path} ---")
        blocks = process_invoice(file_path)
        report = generate_dcp_report(blocks)
        
        print(json.dumps(report, indent=4, ensure_ascii=False))
        
    except Exception as e:
        print(f"Erreur détectée : {e}")