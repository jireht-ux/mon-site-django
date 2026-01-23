import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json
import re
import os

# 1. Configuration et Chargement
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_invoice(pdf_path):
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
    
    # --- CORRECTION ICI : Clipping entre 0 et 1000 ---
    normalized_boxes = []
    for b in raw_boxes:
        normalized_boxes.append([
            max(0, min(1000, int(b[0] * w_scale))),
            max(0, min(1000, int(b[1] * h_scale))),
            max(0, min(1000, int(b[2] * w_scale))),
            max(0, min(1000, int(b[3] * h_scale)))
        ])

    encoding = processor(img, words, boxes=normalized_boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    labels = [model.config.id2label[p] for p in predictions[:len(words)]]

    tokens_info = [{'text': t, 'label': l, 'bbox': b} for t, l, b in zip(words, labels, raw_boxes)]
    
    # --- AJUSTEMENT SEUILS : On réduit pour éviter le bloc unique ---
    def get_clusters(data, threshold_x=20, threshold_y=5):
        clusters = []
        if not data: return clusters
        curr_cluster = [data[0]]
        for i in range(1, len(data)):
            prev, curr = data[i-1]['bbox'], data[i]['bbox']
            # Si même ligne (Y proche) et horizontalement proche
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
    if not blocks: return {}

    merged = []
    skip = set()
    anchor_keywords = ["TOTAL", "TTC", "HT", "TVA", "N°", "DATE", "ÉCHÉANCE", "MONTANT", "FACTURE"]
    
    sorted_blocks = sorted(blocks, key=lambda b: (b['top'], b['left']))

    for i in range(len(sorted_blocks)):
        if i in skip: continue
        # On fait une COPIE pour éviter de modifier l'objet original pendant l'itération
        curr = dict(sorted_blocks[i])
        content_up = curr['content'].upper()
        
        if any(key in content_up for key in anchor_keywords):
            found_h = False
            # Recherche HORIZONTALE (plus courte distance : 0.15 au lieu de balayer toute la ligne)
            for j in range(i + 1, len(sorted_blocks)):
                if j in skip: continue
                cand = sorted_blocks[j]
                
                same_line = abs(curr['top'] - cand['top']) < 0.010
                is_near_right = 0 < (cand['left'] - curr['right']) < 0.20
                
                if same_line and is_near_right:
                    curr['content'] += f" : {cand['content']}"
                    curr['right'] = max(curr['right'], cand['right'])
                    skip.add(j)
                    found_h = True
                    break 

        merged.append(curr)

    dcp_zones = {}
    patterns = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "SIRET": r'\d{14}'
    }

    for i, block in enumerate(merged):
        content = block['content']
        final_label = block['label_IA']
        is_sensitive = False

        for d_name, p in patterns.items():
            if re.search(p, content):
                final_label, is_sensitive = f"ZONE_{d_name}", True
                break
        
        if any(kw in content.upper() for kw in ["SARL", "SAS", "SCT"]):
            final_label = "SOCIETE_EMETTEUR"
        
        if "CLIENT" in content.upper() or is_sensitive:
            is_sensitive = True

        if final_label != "O" or is_sensitive:
            zone_key = f"{final_label}_{i}"
            dcp_zones[zone_key] = {
                "top": block['top'], "content": content, "is_sensitive": is_sensitive
            }

    return dcp_zones

# --- TEST ---
try:
    raw_blocks = process_invoice("facture_23.pdf")
    dcp_report = generate_dcp_report(raw_blocks)
    print(json.dumps(dcp_report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")