import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json
import re
import os

# 1. Chargement des modèles
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_invoice(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    raw_words_data = page.get_text("words")
    if not raw_words_data: return []
    
    # Tri spatial pour une lecture naturelle
    words_data = sorted(raw_words_data, key=lambda w: (w[1], w[0]))
    
    words = [w[4] for w in words_data]
    raw_boxes = [[w[0], w[1], w[2], w[3]] for w in words_data]
    
    # Normalisation 0-1000 avec sécurité
    w_scale, h_scale = 1000 / page_w, 1000 / page_h
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
    
    # CLUSTERING SERRÉ : threshold_y bas pour séparer les lignes
    def get_clusters(data, threshold_x=25, threshold_y=6):
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
        clean_labels = [w['label'].replace("B-","").replace("I-","") for w in cluster if w['label'] != "O"]
        main_label = max(set(clean_labels), key=clean_labels.count) if clean_labels else "O"
        structured_blocks.append({
            "label_IA": main_label,
            "top": round(min(w['bbox'][1] for w in cluster) / page_h, 3),
            "bottom": round(max(w['bbox'][3] for w in cluster) / page_h, 3),
            "left": round(min(w['bbox'][0] for w in cluster) / page_w, 3),
            "right": round(max(w['bbox'][2] for w in cluster) / page_w, 3),
            "content": " ".join([w['text'] for w in cluster])
        })
    return structured_blocks

def generate_dcp_report(blocks):
    if not blocks: return {}
    
    merged = []
    skip = set()
    # Mots-clés déclencheurs de fusion
    anchor_keywords = ["TOTAL", "TTC", "HT", "TVA", "N°", "DATE", "ÉCHÉANCE", "MONTANT", "PU"]
    sorted_blocks = sorted(blocks, key=lambda b: (b['top'], b['left']))

    for i in range(len(sorted_blocks)):
        if i in skip: continue
        curr = dict(sorted_blocks[i]) # Copie pour casser la cascade
        content_up = curr['content'].upper()
        
        if any(key in content_up for key in anchor_keywords):
            target_idx = -1
            # 1. Recherche HORIZONTALE (Priorité 1)
            for j in range(i + 1, len(sorted_blocks)):
                if j in skip: continue
                cand = sorted_blocks[j]
                if abs(curr['top'] - cand['top']) < 0.010 and 0 < (cand['left'] - curr['right']) < 0.2:
                    target_idx = j
                    break 
            
            # 2. Recherche VERTICALE (Si rien à droite)
            if target_idx == -1:
                for j in range(i + 1, len(sorted_blocks)):
                    if j in skip: continue
                    cand = sorted_blocks[j]
                    if 0 <= (cand['top'] - curr['bottom']) < 0.025 and abs(curr['left'] - cand['left']) < 0.05:
                        target_idx = j
                        break
            
            if target_idx != -1:
                target = sorted_blocks[target_idx]
                curr['content'] += f" : {target['content']}"
                curr['right'] = max(curr['right'], target['right'])
                curr['bottom'] = max(curr['bottom'], target['bottom'])
                skip.add(target_idx)

        merged.append(curr)

    # Audit final
    dcp_zones = {}
    patterns = {"EMAIL": r'[\w\.-]+@[\w\.-]+\.\w+', "SIRET": r'\d{14}', "TVA": r'[A-Z]{2}\d{11}'}

    for idx, b in enumerate(merged):
        content = b['content']
        label = b['label_IA']
        is_sens = False

        for d_name, p in patterns.items():
            if re.search(p, content):
                label, is_sens = f"ZONE_{d_name}", True
        
        up = content.upper()
        if "SARL" in up: label = "SOCIETE_EMETTEUR"
        if "FACTURE N°" in up or "DATE" in up: label = f"HEADER_{idx}"
        if "CLIENT" in up: label, is_sens = "QUESTION", True
        if label == "O": label = f"BLOCK_{idx}" # Évite les clés vides

        dcp_zones[f"{label}_{idx}"] = {
            "top": b['top'], "bottom": b['bottom'], "left": b['left'], 
            "right": b['right'], "content": content, "is_sensitive": is_sens
        }
    return dcp_zones

# --- EXECUTION ---
try:
    raw_blocks = process_invoice("facture_2.pdf")
    report = generate_dcp_report(raw_blocks)
    print(json.dumps(report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")