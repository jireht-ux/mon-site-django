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
    """Version avec Verrouillage de fusion pour éviter l'effet cascade."""
    if not blocks: return {}

    merged = []
    skip = set()
    anchor_keywords = ["TOTAL", "TTC", "HT", "TVA", "N°", "DATE", "ÉCHÉANCE", "MONTANT", "FACTURE"]
    
    # Tri par Y (top) puis X (left)
    sorted_blocks = sorted(blocks, key=lambda b: (b['top'], b['left']))

    for i in range(len(sorted_blocks)):
        if i in skip: continue
        curr = sorted_blocks[i].copy() # On travaille sur une copie pour protéger l'original
        content_up = curr['content'].upper()
        
        # On ne tente de fusionner QUE si le bloc actuel est une ancre connue
        if any(key in content_up for key in anchor_keywords):
            target_idx = -1
            
            # --- PRIORITÉ 1 : HORIZONTALE ---
            for j in range(i + 1, len(sorted_blocks)):
                if j in skip: continue
                cand = sorted_blocks[j]
                # Si sur la même ligne et à droite
                if abs(curr['top'] - cand['top']) < 0.012 and cand['left'] > curr['left']:
                    target_idx = j
                    break # On prend le PREMIER à droite et on s'arrête
            
            # --- PRIORITÉ 2 : VERTICALE (si rien trouvé à droite) ---
            if target_idx == -1:
                for j in range(i + 1, len(sorted_blocks)):
                    if j in skip: continue
                    cand = sorted_blocks[j]
                    # Si juste en dessous et aligné (Seuil très serré : 3%)
                    is_below = 0 <= (cand['top'] - curr['bottom']) < 0.03
                    is_aligned = abs(curr['left'] - cand['left']) < 0.08
                    if is_below and is_aligned:
                        target_idx = j
                        break # On prend le PREMIER en dessous et on s'arrête

            # --- EXECUTION DE LA FUSION ---
            if target_idx != -1:
                target_block = sorted_blocks[target_idx]
                curr['content'] += f" : {target_block['content']}"
                curr['right'] = max(curr['right'], target_block['right'])
                curr['bottom'] = max(curr['bottom'], target_block['bottom'])
                skip.add(target_idx) # On marque le bloc trouvé comme utilisé
        
        merged.append(curr)

    # 2. AUDIT FINAL DES DCP
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

        for d_name, p in patterns.items():
            if re.search(p, content):
                final_label = f"ZONE_{d_name}"
                is_sensitive = True
                break
        
        up_content = content.upper()
        if any(kw in up_content for kw in ["SARL", "SAS", "ETABLISSEMENT"]):
            final_label = "SOCIETE_EMETTEUR"
        
        if "CLIENT" in up_content or is_sensitive:
            is_sensitive = True

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