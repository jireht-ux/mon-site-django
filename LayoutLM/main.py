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
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Le fichier {pdf_path} est introuvable.")

    print(f"--- Analyse du document : {pdf_path} ---")
    doc = fitz.open(pdf_path)
    
    if len(doc) == 0:
        raise ValueError("Le document PDF est vide ou corrompu.")
    
    page = doc[0] 
    
    # Conversion en image pour LayoutLMv3
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Extraction et tri des mots
    raw_words_data = page.get_text("words")
    if not raw_words_data:
        print("Avertissement : Aucun texte trouvé sur la page.")
        return {}

    words_data = sorted(raw_words_data, key=lambda w: (w[1], w[0]))
    
    words = [w[4] for w in words_data]
    raw_boxes = [[w[0], w[1], w[2], w[3]] for w in words_data]
    
    # Normalisation des coordonnées (0-1000)
    w_scale, h_scale = 1000 / page.rect.width, 1000 / page.rect.height
    normalized_boxes = [[int(b[0]*w_scale), int(b[1]*h_scale), int(b[2]*w_scale), int(b[3]*h_scale)] for b in raw_boxes]

    # Prédiction IA
    encoding = processor(img, words, boxes=normalized_boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    labels = [model.config.id2label[p] for p in predictions[:len(words)]]

    # Préparation des tokens pour le clustering
    tokens_with_info = []
    for word, label, box in zip(words, labels, raw_boxes):
        tokens_with_info.append({'text': word, 'label': label, 'bbox': box})

    # Algorithme de Clustering Spatial
    def get_clusters(data, threshold_x=45, threshold_y=15):
        clusters = []
        if not data: return clusters
        current_cluster = [data[0]]
        for i in range(1, len(data)):
            prev, curr = data[i-1]['bbox'], data[i]['bbox']
            if abs(curr[1] - prev[1]) < threshold_y and (curr[0] - prev[2]) < threshold_x:
                current_cluster.append(data[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [data[i]]
        clusters.append(current_cluster)
        return clusters

    clusters = get_clusters(tokens_with_info)

    # Analyse et structuration des blocs
    structured_blocks = []
    page_w, page_h = page.rect.width, page.rect.height

    for cluster in clusters:
        full_text = " ".join([w['text'] for w in cluster])
        cluster_labels = [w['label'].replace("B-","").replace("I-","") for w in cluster if w['label'] != "O"]
        main_label = max(set(cluster_labels), key=cluster_labels.count) if cluster_labels else "O"
        
        x0 = min([w['bbox'][0] for w in cluster])
        y0 = min([w['bbox'][1] for w in cluster])
        x1 = max([w['bbox'][2] for w in cluster])
        y1 = max([w['bbox'][3] for w in cluster])

        structured_blocks.append({
            "label_IA": main_label,
            "top": round(y0 / page_h, 3),
            "bottom": round(y1 / page_h, 3),
            "left": round(x0 / page_w, 3),
            "right": round(x1 / page_w, 3),
            "content": full_text
        })

    return structured_blocks

def generate_dcp_report(structured_clusters):
    """
    Transforme les blocs structurés en dictionnaire final DCP_ZONES.
    """
    dcp_zones = {}
    patterns = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "PHONE": r'(?:\+237|237)?\s*[2368]\s*[0-9](?:\s*[0-9]{2}){3}',
        "SIRET": r'\d{14}',
        "TVA": r'[A-Z]{2}\d{11}'
    }

    for i, cluster in enumerate(structured_clusters):
        content = cluster['content']
        label_ia = cluster['label_IA']
        final_type = label_ia
        
        # Détection Regex pour affiner le label
        has_dcp = False
        for dcp_name, pattern in patterns.items():
            if re.search(pattern, content):
                final_type = f"ZONE_{dcp_name}"
                has_dcp = True
                break
        
        # Forcer des types spécifiques sur mots clés
        if "SARL" in content.upper() or "SAS" in content.upper():
            final_type = "SOCIETE_EMETTEUR"

        # On ne garde que les zones pertinentes (IA labels ou présence de DCP)
        if final_type != "O" or has_dcp:
            zone_key = f"{final_type}_{i}"
            dcp_zones[zone_key] = {
                "top": cluster['top'],
                "bottom": cluster['bottom'],
                "left": cluster['left'],
                "right": cluster['right'],
                "content": content,
                "is_sensitive": has_dcp or "CLIENT" in content.upper()
            }
        
    return dcp_zones

# --- EXECUTION ---
pdf_a_tester = "facture_24.pdf"

try:
    # 1. Extraction des blocs par IA + Clustering
    blocs_identifies = process_invoice(pdf_a_tester)
    
    # 2. Génération du rapport final d'audit
    dcp_zones_final = generate_dcp_report(blocs_identifies)

    print("\n--- RAPPORT D'AUDIT DCP GÉNÉRÉ ---")
    print(json.dumps(dcp_zones_final, indent=4, ensure_ascii=False))

except Exception as e:
    print(f"Erreur lors du traitement : {e}")