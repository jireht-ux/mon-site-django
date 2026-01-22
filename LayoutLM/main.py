import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from collections import defaultdict

# 1. Chargement du processeur et du modèle
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_invoice(pdf_path):
    print(f"--- Analyse du document : {pdf_path} ---")
    doc = fitz.open(pdf_path)
    page = doc[0] 
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    words_data = page.get_text("words", sort=True) 
    words = [w[4] for w in words_data]
    
    # Coordonnées brutes pour le calcul des zones plus tard
    raw_boxes = [[w[0], w[1], w[2], w[3]] for w in words_data]
    
    # Normalisation 0-1000 pour LayoutLM
    w_scale = 1000 / page.rect.width
    h_scale = 1000 / page.rect.height
    normalized_boxes = [[int(b[0]*w_scale), int(b[1]*h_scale), int(b[2]*w_scale), int(b[3]*h_scale)] for b in raw_boxes]

    encoding = processor(img, words, boxes=normalized_boxes, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Récupération des prédictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    labels = [model.config.id2label[p] for p in predictions]
    
    # --- LOGIQUE DE REGROUPEMENT ET GÉNÉRATION DCP_ZONES ---
    detected_entities = []
    dcp_zones_dynamic = {}
    
    current_entity = {"text": [], "label": None, "boxes": []}

    print("\n--- ÉTIQUETTES DÉTECTÉES (CONSOLE) ---")
    for word, label, raw_box in zip(words, labels, raw_boxes):
        if label != "O":
            print(f"TOKEN: {word:15} | LABEL: {label}")
            
            # Nettoyage du préfixe B- ou I- pour regrouper
            clean_label = label.replace("B-", "").replace("I-", "")
            
            # Si c'est le début d'une nouvelle entité ou une entité différente
            if label.startswith("B-") or clean_label != current_entity["label"]:
                if current_entity["text"]:
                    detected_entities.append(current_entity)
                current_entity = {"text": [word], "label": clean_label, "boxes": [raw_box]}
            else:
                current_entity["text"].append(word)
                current_entity["boxes"].append(raw_box)
    
    # Ajouter la dernière entité
    if current_entity["text"]:
        detected_entities.append(current_entity)

    # --- CONSTRUCTION DU DCP_ZONES DYNAMIQUE ---
    print("\n--- RÉSUMÉ DES DCP TROUVÉS ---")
    page_width = page.rect.width
    page_height = page.rect.height

    for i, entity in enumerate(detected_entities):
        full_text = " ".join(entity["text"])
        label = entity["label"]
        
        # Calcul de la zone englobante (min/max des boites de mots)
        all_x = [b[0] for b in entity["boxes"]] + [b[2] for b in entity["boxes"]]
        all_y = [b[1] for b in entity["boxes"]] + [b[3] for b in entity["boxes"]]
        
        # Coordonnées relatives (0.0 à 1.0) pour ton Optimizer
        z_top = min(all_y) / page_height
        z_bottom = max(all_y) / page_height
        z_left = min(all_x) / page_width
        z_right = max(all_x) / page_width
        
        # Remplissage du dictionnaire
        zone_key = f"{label}_{i}"
        dcp_zones_dynamic[zone_key] = {
            "top": round(z_top, 3),
            "bottom": round(z_bottom, 3),
            "left": round(z_left, 3),
            "right": round(z_right, 3),
            "content": full_text
        }
        
        print(f"[{label}] : {full_text}")

    return dcp_zones_dynamic

# --- TEST -
# Assure-toi d'avoir un fichier 'facture.pdf' dans le dossier
try:
    result_zones = process_invoice("facture_23.pdf")
    
    print("\n--- DICTIONNAIRE DCP_ZONES GÉNÉRÉ ---")
    import json
    print(json.dumps(result_zones, indent=4))
except Exception as e:
    print(f"Erreur : {e}")