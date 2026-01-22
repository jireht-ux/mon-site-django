import fitz
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json
import re

# 1. Chargement (inchangé)
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_invoice(pdf_path):
    print(f"--- Analyse du document : {pdf_path} ---")
    doc = fitz.open(pdf_path)
    page = doc[0] 
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Correction : Tri manuel si ta version de PyMuPDF est ancienne
    raw_words_data = page.get_text("words")
    words_data = sorted(raw_words_data, key=lambda w: (w[1], w[0]))
    
    words = [w[4] for w in words_data]
    raw_boxes = [[w[0], w[1], w[2], w[3]] for w in words_data]
    
    w_scale, h_scale = 1000 / page.rect.width, 1000 / page.rect.height
    normalized_boxes = [[int(b[0]*w_scale), int(b[1]*h_scale), int(b[2]*w_scale), int(b[3]*h_scale)] for b in raw_boxes]

    encoding = processor(img, words, boxes=normalized_boxes, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    # On gère le cas où LayoutLM crée plus de tokens que de mots (sub-tokens)
    labels = [model.config.id2label[p] for p in predictions[:len(words)]]

    # --- NOUVELLE LOGIQUE : PRÉPARATION DU CLUSTERING ---
    tokens_with_info = []
    for word, label, box in zip(words, labels, raw_boxes):
        tokens_with_info.append({'text': word, 'label': label, 'bbox': box})

    # FONCTION DE CLUSTERING (Distance-based)
    def get_clusters(data, threshold_x=45, threshold_y=15):
        clusters = []
        if not data: return clusters
        current_cluster = [data[0]]
        for i in range(1, len(data)):
            prev, curr = data[i-1]['bbox'], data[i]['bbox']
            # Distance horizontale (fin du mot précédent -> début du mot actuel)
            # Distance verticale (alignement des hauts de ligne)
            if abs(curr[1] - prev[1]) < threshold_y and (curr[0] - prev[2]) < threshold_x:
                current_cluster.append(data[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [data[i]]
        clusters.append(current_cluster)
        return clusters

    clusters = get_clusters(tokens_with_info)

    # --- ANALYSE DES BLOCS (VOTE MAJORITAIRE ET RAFFINAGE) ---
    final_zones = {}
    page_w, page_h = page.rect.width, page.rect.height

    for i, cluster in enumerate(clusters):
        # 1. Texte et Label dominant
        full_text = " ".join([w['text'] for w in cluster])
        cluster_labels = [w['label'].replace("B-","").replace("I-","") for w in cluster if w['label'] != "O"]
        
        # Si le bloc contient des prédictions IA, on prend la plus fréquente, sinon "O"
        main_label = max(set(cluster_labels), key=cluster_labels.count) if cluster_labels else "O"
        
        # 2. Coordonnées du bloc (Boîte englobante)
        x0 = min([w['bbox'][0] for w in cluster])
        y0 = min([w['bbox'][1] for w in cluster])
        x1 = max([w['bbox'][2] for w in cluster])
        y1 = max([w['bbox'][3] for w in cluster])

        # 3. Raffinage par Regex (DCP)
        # On peut forcer un label si une Regex trouve une info critique
        if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', full_text):
            main_label = "CONTACT_EMAIL"
        elif re.search(r'SARL|SAS|EURL|SA ', full_text):
            main_label = "SOCIETE_EMETTEUR"

        # 4. Stockage si le bloc est utile
        if main_label != "O" or "SARL" in full_text:
            zone_key = f"{main_label}_{i}"
            final_zones[zone_key] = {
                "top": round(y0 / page_h, 3),
                "bottom": round(y1 / page_h, 3),
                "left": round(x0 / page_w, 3),
                "right": round(x1 / page_w, 3),
                "content": full_text
            }

    return final_zones

def generate_dcp_report(structured_clusters):
    """
    Analyse les blocs structurés pour créer le dictionnaire final DCP_ZONES
    en isolant les types de données via Regex.
    """
    dcp_zones = {}
    
    # Patterns pour l'identification fine
    patterns = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "PHONE": r'(?:\+237|237)?\s*[2368]\s*[0-9](?:\s*[0-9]{2}){3}',
        "SIRET": r'\d{14}',
        "TVA": r'[A-Z]{2}\d{11}'
    }

    for i, cluster in enumerate(structured_clusters):
        content = cluster['content']
        label_ia = cluster['label_IA']
        
        # Détermination du type de zone final
        final_type = label_ia
        
        # Raffinement stratégique : si une Regex match, on précise le label
        for dcp_name, pattern in patterns.items():
            if re.search(pattern, content):
                final_type = f"ZONE_{dcp_name}"
                break # On prend le premier type de DCP trouvé comme label de zone
        
        # Construction de l'entrée du dictionnaire
        zone_id = f"{final_type}_{i}"
        dcp_zones[zone_id] = {
            "top": cluster['top'],
            "bottom": cluster['bottom'],
            "left": cluster['left'],
            "right": cluster['right'],
            "content": content,
            "is_sensitive": any(re.search(p, content) for p in patterns.values()) or "CLIENT" in content.upper()
        }
        
    return dcp_zones

# --- DANS TON FLUX PRINCIPAL ---
# 1. Tu récupères tes clusters structurés (ceux de ton étape précédente)
# clusters_data = process_invoice("facture_2.pdf") 


# --- EXECUTION ---
try:
    resultat = process_invoice("facture_24.pdf")
    print(json.dumps(resultat, indent=4, ensure_ascii=False))


    # 2. Tu génères le dictionnaire final
    dcp_zones_final = generate_dcp_report(resultat)

    print(json.dumps(dcp_zones_final, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")