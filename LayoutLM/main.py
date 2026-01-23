import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json
import re
import os

# 1. Configuration et Chargement des modèles
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_invoice(pdf_path):
    """Analyse le PDF et extrait des blocs structurés par clustering spatial."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Le fichier {pdf_path} est introuvable.")

    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    raw_words_data = page.get_text("words")
    if not raw_words_data: return []
    words_data = sorted(raw_words_data, key=lambda w: (w[1], w[0]))
    
    words = [w[4] for w in words_data]
    raw_boxes = [[w[0], w[1], w[2], w[3]] for w in words_data]
    
    w_scale, h_scale = 1000 / page_w, 1000 / page_h
    normalized_boxes = [[int(b[0]*w_scale), int(b[1]*h_scale), int(b[2]*w_scale), int(b[3]*h_scale)] for b in raw_boxes]

    encoding = processor(img, words, boxes=normalized_boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    labels = [model.config.id2label[p] for p in predictions[:len(words)]]

    tokens_info = [{'text': t, 'label': l, 'bbox': b} for t, l, b in zip(words, labels, raw_boxes)]
    
    def get_clusters(data, threshold_x=45, threshold_y=12):
        clusters = []
        if not data: return clusters
        curr_cluster = [data[0]]
        for i in range(1, len(data)):
            prev, curr = data[i-1]['bbox'], data[i]['bbox']
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
            "top": round(min(w['bbox'][1] for w in cluster) / page_h, 3),
            "bottom": round(max(w['bbox'][3] for w in cluster) / page_h, 3),
            "left": round(min(w['bbox'][0] for w in cluster) / page_w, 3),
            "right": round(max(w['bbox'][2] for w in cluster) / page_w, 3),
            "content": full_text
        })

    return structured_blocks

def generate_dcp_report(blocks):
    """Fusionne libellés/valeurs puis identifie les zones sans fusionner tout le document."""
    if not blocks: return {}

    merged = []
    skip = set()
    # Mots-clés qui appellent une valeur à côté ou en dessous
    anchor_keywords = ["TOTAL", "TTC", "HT", "TVA", "N°", "DATE", "ÉCHÉANCE", "MONTANT", "FACTURE"]
    
    sorted_blocks = sorted(blocks, key=lambda b: (b['top'], b['left']))

    # 1. PHASE DE FUSION BIDIRECTIONNELLE CIBLÉE
    for i in range(len(sorted_blocks)):
        if i in skip: continue
        curr = sorted_blocks[i]
        content_up = curr['content'].upper()
        
        is_anchor = any(key in content_up for key in anchor_keywords)
        
        if is_anchor:
            found_val = False
            # A. RECHERCHE HORIZONTALE (Prioritaire)
            for j in range(i + 1, len(sorted_blocks)):
                if j in skip: continue
                cand = sorted_blocks[j]
                if abs(curr['top'] - cand['top']) < 0.012 and cand['left'] > curr['left']:
                    curr['content'] += f" : {cand['content']}"
                    curr['right'] = max(curr['right'], cand['right'])
                    skip.add(j)
                    found_val = True
                    break 
            
            # B. RECHERCHE VERTICALE (Si rien à droite)
            if not found_val:
                for j in range(i + 1, len(sorted_blocks)):
                    if j in skip: continue
                    cand = sorted_blocks[j]
                    # Seuil vertical strict (3% de la page) pour éviter la fusion globale
                    is_below = 0 <= (cand['top'] - curr['bottom']) < 0.035
                    is_aligned = abs(curr['left'] - cand['left']) < 0.08
                    if is_below and is_aligned:
                        curr['content'] += f" : {cand['content']}"
                        curr['bottom'] = max(curr['bottom'], cand['bottom'])
                        curr['right'] = max(curr['right'], cand['right'])
                        skip.add(j)
                        break
        
        merged.append(curr)

    # 2. PHASE D'AUDIT (SANS macro-clustering pour éviter le bloc unique)
    dcp_zones = {}
    patterns = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "PHONE": r'(?:\+237|237)?\s*[2368]\s*[0-9](?:\s*[0-9]{2}){3}',
        "SIRET": r'\d{14}',
        "TVA_NUMBER": r'[A-Z]{2}\d{11}'
    }

    for i, block in enumerate(merged):
        content = block['content']
        final_label = block['label_IA']
        is_sensitive = False

        # Detection par Patterns
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

        # On ne garde que les zones pertinentes
        if final_label != "O" or is_sensitive:
            zone_key = f"{final_label}_{i}"
            dcp_zones[zone_key] = {
                "top": block['top'], "bottom": block['bottom'],
                "left": block['left'], "right": block['right'],
                "content": content, "is_sensitive": is_sensitive
            }

    return dcp_zones

# --- EXECUTION ---
try:
    raw_blocks = process_invoice("facture_2.pdf")
    dcp_report = generate_dcp_report(raw_blocks)
    print(json.dumps(dcp_report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")