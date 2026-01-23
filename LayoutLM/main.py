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
    if len(doc) == 0:
        raise ValueError("Le document PDF est vide ou corrompu.")
    
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

def cluster_macro_zones(blocks, v_threshold=0.06):
    """
    Regroupe les blocs fusionnés en Macro-Zones sémantiques basées sur la proximité.
    """
    macro_zones = []
    # Tri par axe vertical pour regrouper les paragraphes
    sorted_blocks = sorted(blocks, key=lambda b: b['top'])
    
    if not sorted_blocks: return []

    current_zone = [sorted_blocks[0]]
    
    for i in range(1, len(sorted_blocks)):
        prev = current_zone[-1]
        curr = sorted_blocks[i]
        
        # Calcul de la distance verticale entre le bas du bloc précédent et le haut du suivant
        dist_y = curr['top'] - prev['bottom']
        # Calcul de l'alignement horizontal (si les blocs partagent une zone commune sur X)
        overlap_x = min(prev['right'], curr['right']) - max(prev['left'], curr['left'])
        
        # Si les blocs sont proches verticalement et alignés, ils font partie du même paragraphe/zone
        if dist_y < v_threshold and (overlap_x > -0.05):
            current_zone.append(curr)
        else:
            macro_zones.append(current_zone)
            current_zone = [curr]
    
    macro_zones.append(current_zone)
    return macro_zones

def generate_dcp_report(blocks):
    """Analyse les blocs, fusionne les ancres, puis crée des macro-zones thématiques."""
    if not blocks: return {}

    # --- ÉTAPE 1 : FUSION DES ANCRES (LIAISON LIBELLÉ:VALEUR) ---
    merged = []
    skip = set()
    anchor_keywords = ["TOTAL", "TTC", "HT", "TVA", "N°", "DATE", "ÉCHÉANCE", "MONTANT", "FACTURE"]
    sorted_blocks = sorted(blocks, key=lambda b: (b['top'], b['left']))

    for i in range(len(sorted_blocks)):
        if i in skip: continue
        curr = sorted_blocks[i]
        content_up = curr['content'].upper()
        
        if any(key in content_up for key in anchor_keywords):
            found_h = False
            for j in range(i + 1, len(sorted_blocks)):
                if j in skip: continue
                cand = sorted_blocks[j]
                if abs(curr['top'] - cand['top']) < 0.015 and cand['left'] > curr['left']:
                    curr['content'] += f" : {cand['content']}"
                    curr['right'] = max(curr['right'], cand['right'])
                    skip.add(j)
                    found_h = True
                    break 
            
            if not found_h:
                for j in range(i + 1, len(sorted_blocks)):
                    if j in skip: continue
                    cand = sorted_blocks[j]
                    is_below = 0 <= (cand['top'] - curr['bottom']) < 0.04
                    curr_mid_x = (curr['left'] + curr['right']) / 2
                    cand_mid_x = (cand['left'] + cand['right']) / 2
                    if is_below and abs(curr_mid_x - cand_mid_x) < 0.12:
                        curr['content'] += f" : {cand['content']}"
                        curr['bottom'] = max(curr['bottom'], cand['bottom'])
                        curr['right'] = max(curr['right'], cand['right'])
                        skip.add(j)
                        break
        merged.append(curr)

    # --- ÉTAPE 2 : CRÉATION DES MACRO-ZONES ---
    macro_groups = cluster_macro_zones(merged)
    
    final_report = {}
    patterns = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "PHONE": r'(?:\+237|237)?\s*[2368]\s*[0-9](?:\s*[0-9]{2}){3}',
        "SIRET": r'\d{14}',
        "TVA_NUMBER": r'[A-Z]{2}\d{11}'
    }

    for idx, group in enumerate(macro_groups):
        # On compile tout le texte de la zone
        full_zone_content = " | ".join([b['content'] for b in group])
        
        # Coordonnées de la Macro-Zone (Bounding Box englobante)
        z_top = min(b['top'] for b in group)
        z_bottom = max(b['bottom'] for b in group)
        z_left = min(b['left'] for b in group)
        z_right = max(b['right'] for b in group)
        
        # Qualification sémantique
        zone_type = "GENERAL_INFO"
        is_sensitive = False
        
        # Test de sensibilité
        for d_name, p in patterns.items():
            if re.search(p, full_zone_content):
                zone_type = f"MACRO_ZONE_{d_name}"
                is_sensitive = True
        
        if any(kw in full_zone_content.upper() for kw in ["CLIENT", "FACTURÉ À"]):
            zone_type = "MACRO_ZONE_CLIENT"
            is_sensitive = True
        elif any(kw in full_zone_content.upper() for kw in ["TOTAL", "TTC", "HT", "NET"]):
            zone_type = "MACRO_ZONE_FINANCIAL"
        elif any(kw in full_zone_content.upper() for kw in ["SARL", "SAS", "ETABLISSEMENT"]):
            zone_type = "MACRO_ZONE_EMETTEUR"

        final_report[f"ZONE_{idx}"] = {
            "type": zone_type,
            "coordinates": {"top": z_top, "bottom": z_bottom, "left": z_left, "right": z_right},
            "is_sensitive": is_sensitive,
            "content": full_zone_content,
            "blocks_count": len(group)
        }

    return final_report

# --- EXECUTION ---
try:
    raw_blocks = process_invoice("facture_2.pdf")
    final_report = generate_dcp_report(raw_blocks)
    print(json.dumps(final_report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")