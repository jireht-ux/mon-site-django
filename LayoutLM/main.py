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
    
    # Extraction brute des mots
    raw_words_data = page.get_text("words")
    if not raw_words_data: return []

    # TRI PAR LIGNE STRICT (Tolérance 3 pixels)
    # On trie d'abord par Y, puis par X
    words_data = sorted(raw_words_data, key=lambda w: (w[1], w[0]))
    
    # Reconstruction des lignes physiques
    lines = []
    if words_data:
        current_line = [words_data[0]]
        for i in range(1, len(words_data)):
            prev_w = words_data[i-1]
            curr_w = words_data[i]
            # Si l'écart vertical est > 3px, c'est une nouvelle ligne
            if abs(curr_w[1] - prev_w[1]) > 3:
                lines.append(current_line)
                current_line = [curr_w]
            else:
                current_line.append(curr_w)
        lines.append(current_line)

    # Préparation pour LayoutLMv3
    all_words = []
    all_boxes = []
    for line in lines:
        for w in line:
            all_words.append(w[4])
            all_boxes.append([w[0], w[1], w[2], w[3]])

    # Normalisation 0-1000 sécurisée
    w_scale, h_scale = 1000 / page_w, 1000 / page_h
    normalized_boxes = [[max(0, min(1000, int(b[0]*w_scale))), max(0, min(1000, int(b[1]*h_scale))),
                         max(0, min(1000, int(b[2]*w_scale))), max(0, min(1000, int(b[3]*h_scale)))] for b in all_boxes]

    # Inférence IA
    encoding = processor(img, all_words, boxes=normalized_boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    labels = [model.config.id2label[p] for p in predictions[:len(all_words)]]

    # Création des blocs finaux (un par segment de ligne)
    structured_blocks = []
    for line in lines:
        # On peut encore découper horizontalement si l'espace est grand (>40px)
        line = sorted(line, key=lambda w: w[0])
        temp_cluster = [line[0]]
        for i in range(1, len(line)):
            if (line[i][0] - line[i-1][2]) > 40:
                structured_blocks.append(temp_cluster)
                temp_cluster = [line[i]]
            else:
                temp_cluster.append(line[i])
        structured_blocks.append(temp_cluster)

    final_blocks = []
    for cluster in structured_blocks:
        content = " ".join([w[4] for w in cluster])
        final_blocks.append({
            "top": round(min(w[1] for w in cluster) / page_h, 3),
            "bottom": round(max(w[3] for w in cluster) / page_h, 3),
            "left": round(min(w[0] for w in cluster) / page_w, 3),
            "right": round(max(w[2] for w in cluster) / page_w, 3),
            "content": content,
            "label_IA": "O" # Label par défaut, sera affiné par motifs
        })

    return final_blocks

def generate_dcp_report(blocks):
    if not blocks: return {}
    
    dcp_zones = {}
    patterns = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "SIRET": r'\d{14}',
        "PHONE": r'(?:\+237|237)?\s*[2368]\s*[0-9](?:\s*[0-9]{2}){3}'
    }

    for i, block in enumerate(blocks):
        content = block['content']
        up_content = content.upper()
        label = "BLOCK"
        is_sensitive = False

        # Detection des types
        for d_name, p in patterns.items():
            if re.search(p, content):
                label, is_sensitive = f"ZONE_{d_name}", True
        
        if any(kw in up_content for kw in ["SARL", "SCT", "SOCIETE"]): label = "EMETTEUR"
        if "CLIENT" in up_content: label, is_sensitive = "CLIENT_INFO", True
        if any(kw in up_content for kw in ["FACTURE N°", "DATE", "ÉCHÉANCE"]): label = "HEADER"

        # On ne garde que les zones pertinentes ou sensibles
        if label != "BLOCK" or is_sensitive:
            dcp_zones[f"{label}_{i}"] = {
                "top": block['top'],
                "bottom": block['bottom'],
                "left": block['left'],
                "right": block['right'],
                "content": content,
                "is_sensitive": is_sensitive
            }

    return dcp_zones

# --- TEST ---
try:
    # Remplacez par votre fichier
    raw_blocks = process_invoice("facture_FAC-2025-00001_2.pdf")
    dcp_report = generate_dcp_report(raw_blocks)
    print(json.dumps(dcp_report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")