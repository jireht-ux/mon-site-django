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
    
    # Récupération des mots
    words_data = page.get_text("words") 
    if not words_data: return []

    # --- ÉTAPE CLÉ : DÉCOUPAGE PAR LIGNES PHYSIQUES ---
    # On trie d'abord strictement par le haut (y0)
    words_data = sorted(words_data, key=lambda w: w[1])
    
    lines = []
    if words_data:
        current_line = [words_data[0]]
        for i in range(1, len(words_data)):
            prev_w = words_data[i-1]
            curr_w = words_data[i]
            
            # Si l'écart vertical est > 3 pixels, c'est une nouvelle ligne
            if abs(curr_w[1] - prev_w[1]) > 3:
                lines.append(current_line)
                current_line = [curr_w]
            else:
                current_line.append(curr_w)
        lines.append(current_line)

    # Reconstruction des tokens pour l'IA
    final_tokens = []
    for line in lines:
        # Trier chaque ligne de gauche à droite
        line = sorted(line, key=lambda w: w[0])
        for w in line:
            final_tokens.append({'bbox': [w[0], w[1], w[2], w[3]], 'text': w[4]})

    words = [t['text'] for t in final_tokens]
    raw_boxes = [t['bbox'] for t in final_tokens]
    
    # Normalisation 0-1000
    w_scale, h_scale = 1000 / page_w, 1000 / page_h
    norm_boxes = [[max(0, min(1000, int(b[0]*w_scale))), max(0, min(1000, int(b[1]*h_scale))),
                   max(0, min(1000, int(b[2]*w_scale))), max(0, min(1000, int(b[3]*h_scale)))] for b in raw_boxes]

    # Prediction IA
    encoding = processor(img, words, boxes=norm_boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    
    labels = [model.config.id2label[p] for p in outputs.logits.argmax(-1).squeeze().tolist()[:len(words)]]
    
    # --- CLUSTERING HORIZONTAL UNIQUEMENT ---
    structured_blocks = []
    for line in lines:
        line = sorted(line, key=lambda w: w[0])
        # On crée un bloc par segment de ligne
        temp_cluster = [line[0]]
        for i in range(1, len(line)):
            # Si l'espace horizontal est trop grand (> 30px), on coupe la ligne en deux blocs
            if (line[i][0] - line[i-1][2]) > 30:
                structured_blocks.append(temp_cluster)
                temp_cluster = [line[i]]
            else:
                temp_cluster.append(line[i])
        structured_blocks.append(temp_cluster)

    # Formatage final
    final_blocks = []
    for cluster in structured_blocks:
        txt = " ".join([w[4] for w in cluster])
        final_blocks.append({
            "top": round(min(w[1] for w in cluster) / page_h, 3),
            "bottom": round(max(w[3] for w in cluster) / page_h, 3),
            "left": round(min(w[0] for w in cluster) / page_w, 3),
            "right": round(max(w[2] for w in cluster) / page_w, 3),
            "content": txt,
            "label_IA": "O" # On simplifie pour le test
        })
    return final_blocks

def generate_dcp_report(blocks):
    dcp_report = {}
    patterns = {"EMAIL": r'[\w\.-]+@[\w\.-]+\.\w+', "SIRET": r'\d{14}'}
    
    for i, b in enumerate(blocks):
        content = b['content']
        is_sens = False
        label = "BLOCK"
        
        for d_name, p in patterns.items():
            if re.search(p, content): 
                label, is_sens = f"ZONE_{d_name}", True
        
        up = content.upper()
        if "SARL" in up: label = "EMETTEUR"
        if "CLIENT" in up: is_sens = True
        if "FACTURE N°" in up: label = "HEADER"

        dcp_report[f"{label}_{i}"] = {
            "top": b['top'], "content": content, "is_sensitive": is_sens
        }
    return dcp_report

# --- EXECUTION ---
if __name__ == "__main__":
    blocks = process_invoice("facture_2.pdf")
    report = generate_dcp_report(blocks)
    print(json.dumps(report, indent=4, ensure_ascii=False))