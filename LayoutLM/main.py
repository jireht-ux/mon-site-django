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
    
    # Image pour le modèle
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Extraction brute des mots
    words_raw = page.get_text("words") # [x0, y0, x1, y1, "text", ...]
    if not words_raw: return []

    # --- ÉTAPE 1 : DÉCOUPAGE PAR LIGNES STRICTES ---
    # On trie d'abord par le haut de la boîte (y0)
    words_sorted = sorted(words_raw, key=lambda w: w[1])
    
    lines = []
    if words_sorted:
        current_line = [words_sorted[0]]
        for i in range(1, len(words_sorted)):
            prev = words_sorted[i-1]
            curr = words_sorted[i]
            
            # Si l'écart vertical est supérieur à 3 pixels, on change de ligne
            if abs(curr[1] - prev[1]) > 3:
                lines.append(current_line)
                current_line = [curr]
            else:
                current_line.append(curr)
        lines.append(current_line)

    # --- ÉTAPE 2 : SEGMENTATION HORIZONTALE ET IA ---
    structured_blocks = []
    
    for line_index, line in enumerate(lines):
        # Trier chaque ligne de gauche à droite
        line = sorted(line, key=lambda w: w[0])
        
        # On découpe la ligne si un grand espace vide existe (ex: entre libellé et montant)
        temp_cluster = [line[0]]
        clusters_in_line = []
        for i in range(1, len(line)):
            # Si espace > 40 pixels, on crée un nouveau bloc sur la même ligne
            if (line[i][0] - line[i-1][2]) > 40:
                clusters_in_line.append(temp_cluster)
                temp_cluster = [line[i]]
            else:
                temp_cluster.append(line[i])
        clusters_in_line.append(temp_cluster)

        # Pour chaque bloc identifié, on prépare le résultat
        for cluster in clusters_in_line:
            content = " ".join([w[4] for w in cluster])
            
            # Normalisation des coordonnées pour l'IA (0-1000)
            w_scale, h_scale = 1000 / page_w, 1000 / page_h
            
            # On prend le premier mot pour la prédiction de label (simplification)
            first_w = cluster[0]
            norm_bbox = [
                max(0, min(1000, int(first_w[0] * w_scale))),
                max(0, min(1000, int(first_w[1] * h_scale))),
                max(0, min(1000, int(first_w[2] * w_scale))),
                max(0, min(1000, int(first_w[3] * h_scale)))
            ]
            
            # Appel rapide au modèle pour ce petit bloc
            encoding = processor(img, [first_w[4]], boxes=[norm_bbox], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**encoding)
            pred = outputs.logits.argmax(-1).squeeze().tolist()
            # Sécurité si pred est un entier seul
            if isinstance(pred, int): label_id = pred 
            else: label_id = pred[0] if len(pred) > 0 else 0
                
            label_ia = model.config.id2label[label_id]

            structured_blocks.append({
                "top": round(min(w[1] for w in cluster) / page_h, 3),
                "bottom": round(max(w[3] for w in cluster) / page_h, 3),
                "left": round(min(w[0] for w in cluster) / page_w, 3),
                "right": round(max(w[2] for w in cluster) / page_w, 3),
                "content": content,
                "label_ia": label_ia
            })

    return structured_blocks

def generate_dcp_report(blocks):
    """Analyse finale et détection des données sensibles."""
    report = {}
    patterns = {
        "EMAIL": r'[\w\.-]+@[\w\.-]+\.\w+',
        "SIRET": r'\d{14}',
        "TVA": r'[A-Z]{2}\d{11}'
    }

    for i, b in enumerate(blocks):
        content = b['content']
        label = b['label_ia'].replace("B-","").replace("I-","")
        is_sensitive = False

        # Detection par motifs
        for d_name, p in patterns.items():
            if re.search(p, content):
                label = f"ZONE_{d_name}"
                is_sensitive = True
        
        up = content.upper()
        if "SARL" in up or "SAS" in up: label = "SOCIETE_EMETTEUR"
        if "CLIENT" in up: is_sensitive = True
        if "FACTURE N°" in up: label = "HEADER"

        # On crée une clé unique
        key = f"{label}_{i}"
        report[key] = {
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
        path = "facture_2.pdf" # Vérifie bien le nom
        raw_data = process_invoice(path)
        final_json = generate_dcp_report(raw_data)
        print(json.dumps(final_json, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"Erreur : {e}")