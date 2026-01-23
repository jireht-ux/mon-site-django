import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json
import re
import os

# 1. Configuration des modèles
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_invoice(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Extraction des mots avec coordonnées
    words_raw = page.get_text("words") # format: (x0, y0, x1, y1, "text", ...)
    if not words_raw: return []

    # --- ÉTAPE 1 : TRI ET DÉCOUPAGE PAR LIGNES ---
    # On arrondit l'ordonnée (y) pour regrouper les mots qui sont sur la même ligne visuelle
    # tolerance = 3 pixels
    words_sorted = sorted(words_raw, key=lambda w: (round(w[1] / 3), w[0]))
    
    structured_blocks = []
    if words_sorted:
        current_block = {
            "top": words_sorted[0][1],
            "bottom": words_sorted[0][3],
            "left": words_sorted[0][0],
            "right": words_sorted[0][2],
            "content": words_sorted[0][4]
        }
        
        for i in range(1, len(words_sorted)):
            prev_w = words_sorted[i-1]
            curr_w = words_sorted[i]
            
            # CONDITION DE FUSION : Même ligne (écart y < 4) ET proximité horizontale (écart x < 30)
            is_same_line = abs(curr_w[1] - prev_w[1]) < 4
            is_near_h = (curr_w[0] - prev_w[2]) < 30
            
            if is_same_line and is_near_h:
                # On étend le bloc actuel
                current_block["content"] += " " + curr_w[4]
                current_block["right"] = max(current_block["right"], curr_w[2])
                current_block["bottom"] = max(current_block["bottom"], curr_w[3])
            else:
                # On enregistre le bloc et on en commence un nouveau
                structured_blocks.append(current_block)
                current_block = {
                    "top": curr_w[1],
                    "bottom": curr_w[3],
                    "left": curr_w[0],
                    "right": curr_w[2],
                    "content": curr_w[4]
                }
        structured_blocks.append(current_block)

    # --- ÉTAPE 2 : NORMALISATION ET IA ---
    w_scale, h_scale = 1000 / page_w, 1000 / page_h
    final_data = []

    for block in structured_blocks:
        # Prediction IA simplifiée par bloc
        norm_bbox = [
            max(0, min(1000, int(block["left"] * w_scale))),
            max(0, min(1000, int(block["top"] * h_scale))),
            max(0, min(1000, int(block["right"] * w_scale))),
            max(0, min(1000, int(block["bottom"] * h_scale)))
        ]
        
        encoding = processor(img, [block["content"][:10]], boxes=[norm_bbox], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)
        
        label_id = outputs.logits.argmax(-1).squeeze().tolist()
        if isinstance(label_id, list): label_id = label_id[0]
        label_ia = model.config.id2label[label_id]

        final_data.append({
            "top": round(block["top"] / page_h, 3),
            "bottom": round(block["bottom"] / page_h, 3),
            "left": round(block["left"] / page_w, 3),
            "right": round(block["right"] / page_w, 3),
            "content": block["content"],
            "label_ia": label_ia
        })

    return final_data

def generate_dcp_report(blocks):
    report = {}
    patterns = {"EMAIL": r'[\w\.-]+@[\w\.-]+\.\w+', "SIRET": r'\d{14}'}

    for i, b in enumerate(blocks):
        content = b['content']
        label = b['label_ia'].replace("B-","").replace("I-","")
        is_sensitive = False

        for d_name, p in patterns.items():
            if re.search(p, content):
                label, is_sensitive = f"ZONE_{d_name}", True
        
        up = content.upper()
        if "SARL" in up: label = "SOCIETE_EMETTEUR"
        if "CLIENT" in up: is_sensitive = True
        if "FACTURE N°" in up: label = "HEADER"

        report[f"{label}_{i}"] = {
            "top": b['top'],
            "bottom": b['bottom'],
            "left": b['left'],
            "right": b['right'],
            "content": content,
            "is_sensitive": is_sensitive
        }
    return report

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        raw_blocks = process_invoice("facture_2.pdf")
        final_report = generate_dcp_report(raw_blocks)
        print(json.dumps(final_report, indent=4, ensure_ascii=False))
        print(f"test final Total zones extraites : {len(final_report)}")
    except Exception as e:
        print(f"Erreur : {e}")