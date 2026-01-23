import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json
import re

# 1. Initialisation
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_hybrid_invoice(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 2. Extraction des segments physiques
    blocks_raw = page.get_text("dict")["blocks"]
    segments = []
    for b in blocks_raw:
        if "lines" not in b: continue
        for l in b["lines"]:
            for span in l["spans"]:
                txt = span["text"].strip()
                if not txt: continue
                segments.append({
                    "bbox": list(span["bbox"]),
                    "content": txt,
                    "label_ia": "O",
                    "assigned": False
                })

    # 3. Phase d'étiquetage par l'IA
    words = [s["content"] for s in segments]
    boxes = [[max(0, min(1000, int(s["bbox"][0] * (1000/page_w)))),
              max(0, min(1000, int(s["bbox"][1] * (1000/page_h)))),
              max(0, min(1000, int(s["bbox"][2] * (1000/page_w)))),
              max(0, min(1000, int(s["bbox"][3] * (1000/page_h))))] for s in segments]

    encoding = processor(img, words, boxes=boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    for i, p_idx in enumerate(predictions[:len(segments)]):
        segments[i]["label_ia"] = model.config.id2label[p_idx].replace("B-","").replace("I-","")

    # --- LISTE TEMPORAIRE POUR LE REGROUPEMENT ---
    temp_results = []

    # 4. Phase de recherche spatiale (Ancres Clés-Valeurs)
    labels_to_complete = ["HEADER", "QUESTION", "LABEL"] 
    keywords = ["TVA", "TOTAL", "TTC", "HT", "FACTURE", "DATE", "ÉCHÉANCE", "N°"]

    for i, seg in enumerate(segments):
        if seg["assigned"]: continue
        content_up = seg["content"].upper()
        is_anchor = seg["label_ia"] in labels_to_complete or any(k in content_up for k in keywords)

        if is_anchor:
            current_bbox = list(seg["bbox"])
            valeur_associee = ""
            
            # Recherche horizontale puis verticale
            for j in range(i + 1, len(segments)):
                cand = segments[j]
                if cand["assigned"]: continue
                same_line = abs(seg["bbox"][1] - cand["bbox"][1]) < 10 
                at_right = cand["bbox"][0] > (seg["bbox"][0]) 
                
                if same_line and at_right:
                    valeur_associee = cand["content"]
                    current_bbox[2] = max(current_bbox[2], cand["bbox"][2])
                    current_bbox[3] = max(current_bbox[3], cand["bbox"][3])
                    cand["assigned"] = True
                    break
            
            if not valeur_associee:
                for j in range(i + 1, len(segments)):
                    cand = segments[j]
                    if cand["assigned"]: continue
                    dist_v = cand["bbox"][1] - seg["bbox"][3]
                    is_aligned = abs(seg["bbox"][0] - cand["bbox"][0]) < 60
                    if 0 < dist_v < 30 and is_aligned:
                        valeur_associee = cand["content"]
                        current_bbox[2] = max(current_bbox[2], cand["bbox"][2])
                        current_bbox[3] = max(current_bbox[3], cand["bbox"][3])
                        cand["assigned"] = True
                        break

            temp_results.append({
                "label": seg['label_ia'] if seg['label_ia'] != "O" else "DATA",
                "content": f"{seg['content']} {valeur_associee}".strip(),
                "bbox": current_bbox
            })
            seg["assigned"] = True

    # 5. Phase de Regroupement en Macro-Zones (Post-Clustering)
    # On traite les segments restants non-assignés (adresses, émetteur, etc.)
    final_report = {}
    remaining_segs = [s for s in segments if not s["assigned"]]
    
    # On ajoute les résultats des ancres à la liste des objets à clusteriser
    # car ils peuvent aussi faire partie d'un bloc plus large (ex: Date dans Header)
    objects_to_cluster = temp_results + [{"label": s["label_ia"], "content": s["content"], "bbox": s["bbox"]} for s in remaining_segs]
    objects_to_cluster.sort(key=lambda x: x["bbox"][1]) # Tri par top

    macro_clusters = []
    if objects_to_cluster:
        curr_cluster = [objects_to_cluster[0]]
        for i in range(1, len(objects_to_cluster)):
            prev = curr_cluster[-1]
            curr = objects_to_cluster[i]
            
            # SEUIL DE PROXIMITÉ : 15 pixels verticaux pour former une macro-zone
            v_dist = curr["bbox"][1] - prev["bbox"][3]
            h_dist = abs(curr["bbox"][0] - prev["bbox"][0])

            if v_dist < 15 and h_dist < 300: # 300 pour accepter les colonnes gauche/droite proches
                curr_cluster.append(curr)
            else:
                macro_clusters.append(curr_cluster)
                curr_cluster = [curr]
        macro_clusters.append(curr_cluster)

    # 6. Génération du JSON Final
    for idx, cluster in enumerate(macro_clusters):
        # Fusion des textes et des coordonnées du cluster
        combined_text = " ".join([c["content"] for c in cluster])
        min_x = min(c["bbox"][0] for c in cluster)
        min_y = min(c["bbox"][1] for c in cluster)
        max_x = max(c["bbox"][2] for c in cluster)
        max_y = max(c["bbox"][3] for c in cluster)
        
        # On détermine le label dominant du cluster
        labels_in_cluster = [c["label"] for c in cluster if c["label"] != "O"]
        main_label = labels_in_cluster[0] if labels_in_cluster else "ZONE"
        
        final_report[f"{main_label}_{idx}"] = {
            "content": combined_text,
            "top": round(min_y / page_h, 3),
            "bottom": round(max_y / page_h, 3),
            "left": round(min_x / page_w, 3),
            "right": round(max_x / page_w, 3),
            "is_sensitive": any(x in combined_text.upper() for x in ["@", "CLIENT", "TEL"])
        }

    return final_report

# --- EXECUTION ---
try:
    report = process_hybrid_invoice("facture_2.pdf")
    print(json.dumps(report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")