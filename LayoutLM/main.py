import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json
import re
import os

# 1. Configuration
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
        structured_blocks.append({
            "label_IA": max(set([w['label'].replace("B-","").replace("I-","") for w in cluster]), key=lambda x: x) if cluster else "O",
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
    anchor_keywords = ["TOTAL", "TTC", "HT", "TVA", "N°", "DATE", "ÉCHÉANCE", "MONTANT", "FACTURE"]
    
    # Tri spatial strict : Haut -> Bas
    sorted_blocks = sorted(blocks, key=lambda b: (b['top'], b['left']))

    for i in range(len(sorted_blocks)):
        if i in skip: continue
        
        # On travaille sur une COPIE figée pour éviter que le bloc ne "grandisse" et aspire ses voisins
        curr = dict(sorted_blocks[i])
        content_up = curr['content'].upper()
        
        if any(key in content_up for key in anchor_keywords):
            target_idx = -1
            
            # 1. RECHERCHE HORIZONTALE (Une seule cible max)
            for j in range(i + 1, len(sorted_blocks)):
                if j in skip: continue
                cand = sorted_blocks[j]
                if abs(curr['top'] - cand['top']) < 0.012 and cand['left'] > curr['left']:
                    target_idx = j
                    break # ARRET IMMEDIAT : on a trouvé la valeur à droite
            
            # 2. RECHERCHE VERTICALE (Uniquement si rien à droite)
            if target_idx == -1:
                for j in range(i + 1, len(sorted_blocks)):
                    if j in skip: continue
                    cand = sorted_blocks[j]
                    # Seuil strict : 3% de la page vers le bas
                    if 0 <= (cand['top'] - curr['bottom']) < 0.03 and abs(curr['left'] - cand['left']) < 0.06:
                        target_idx = j
                        break # ARRET IMMEDIAT : on a trouvé la valeur en dessous
            
            if target_idx != -1:
                val_block = sorted_blocks[target_idx]
                curr['content'] += f" : {val_block['content']}"
                curr['right'] = max(curr['right'], val_block['right'])
                curr['bottom'] = max(curr['bottom'], val_block['bottom'])
                skip.add(target_idx)

        merged.append(curr)

    # AUDIT ET FILTRAGE
    dcp_report = {}
    patterns = {"EMAIL": r'[\w\.-]+@[\w\.-]+\.\w+', "PHONE": r'(?:\+237|237)?\s*[2368]\s*[0-9](?:\s*[0-9]{2}){3}'}

    for i, block in enumerate(merged):
        content = block['content']
        label = "INFO"
        is_sensitive = False
        
        for d_name, p in patterns.items():
            if re.search(p, content):
                label, is_sensitive = f"ZONE_{d_name}", True
        
        if any(kw in content.upper() for kw in ["SARL", "SAS", "ETABLISSEMENT"]): label = "SOCIETE_EMETTEUR"
        if "CLIENT" in content.upper() or is_sensitive: is_sensitive = True

        if label != "INFO" or is_sensitive:
            dcp_report[f"{label}_{i}"] = {
                "top": block['top'], "content": content, "is_sensitive": is_sensitive
            }
    return dcp_report

# --- EXECUTION ---
try:
    raw_blocks = process_invoice("facture_2.pdf")
    dcp_report = generate_dcp_report(raw_blocks)
    print(json.dumps(dcp_report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")