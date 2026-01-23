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
    
    # 2. Extraction des segments physiques (spans)
    blocks_raw = page.get_text("dict")["blocks"]
    segments = []
    for b in blocks_raw:
        if "lines" not in b: continue
        for l in b["lines"]:
            for span in l["spans"]:
                txt = span["text"].strip()
                if not txt: continue
                segments.append({
                    "bbox": list(span["bbox"]), # [x0, y0, x1, y1]
                    "content": txt,
                    "label_ia": "O",
                    "assigned": False
                })

    # 3. Phase d'étiquetage par l'IA
    words = [s["content"] for s in segments]
    boxes = []
    for s in segments:
        b = s["bbox"]
        boxes.append([
            max(0, min(1000, int(b[0] * (1000/page_w)))),
            max(0, min(1000, int(b[1] * (1000/page_h)))),
            max(0, min(1000, int(b[2] * (1000/page_w)))),
            max(0, min(1000, int(b[3] * (1000/page_h))))
        ])

    encoding = processor(img, words, boxes=boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    for i, p_idx in enumerate(predictions[:len(segments)]):
        segments[i]["label_ia"] = model.config.id2label[p_idx].replace("B-","").replace("I-","")

    # 4. Phase de recherche spatiale (Ancres IA + Géométrie)
    final_report = {}
    
    # Déclencheurs d'ancres
    labels_to_complete = ["HEADER", "QUESTION", "LABEL"] 
    keywords = ["TVA", "TOTAL", "TTC", "HT", "FACTURE", "DATE", "ÉCHÉANCE", "N°"]

    for i, seg in enumerate(segments):
        if seg["assigned"]: continue
        
        content_up = seg["content"].upper()
        is_anchor = seg["label_ia"] in labels_to_complete or any(k in content_up for k in keywords)

        if is_anchor:
            # On initialise les coordonnées de la zone avec l'ancre elle-même
            current_bbox = list(seg["bbox"])
            valeur_associee = ""
            
            # --- Recherche HORIZONTALE (Priorité 1) ---
            for j in range(i + 1, len(segments)):
                cand = segments[j]
                if cand["assigned"]: continue
                
                # Même ligne et situé à droite
                same_line = abs(seg["bbox"][1] - cand["bbox"][1]) < 10 
                at_right = cand["bbox"][0] > (seg["bbox"][0]) 
                
                if same_line and at_right:
                    valeur_associee = cand["content"]
                    # Extension de la bbox vers la droite
                    current_bbox[2] = max(current_bbox[2], cand["bbox"][2])
                    current_bbox[3] = max(current_bbox[3], cand["bbox"][3])
                    cand["assigned"] = True
                    break
            
            # --- Recherche VERTICALE (Priorité 2, si rien trouvé à droite) ---
            if not valeur_associee:
                for j in range(i + 1, len(segments)):
                    cand = segments[j]
                    if cand["assigned"]: continue
                    
                    # En dessous (distance max 30px) et aligné horizontalement
                    dist_v = cand["bbox"][1] - seg["bbox"][3]
                    is_aligned = abs(seg["bbox"][0] - cand["bbox"][0]) < 60
                    
                    if 0 < dist_v < 30 and is_aligned:
                        valeur_associee = cand["content"]
                        # Extension de la bbox vers le bas
                        current_bbox[2] = max(current_bbox[2], cand["bbox"][2])
                        current_bbox[3] = max(current_bbox[3], cand["bbox"][3])
                        cand["assigned"] = True
                        break

            # Construction du dictionnaire final
            label_final = seg['label_ia'] if seg['label_ia'] != "O" else "DATA"
            final_report[f"{label_final}_{i}"] = {
                "content": f"{seg['content']} {valeur_associee}".strip(),
                "top": round(current_bbox[1] / page_h, 3),
                "bottom": round(current_bbox[3] / page_h, 3),
                "left": round(current_bbox[0] / page_w, 3),
                "right": round(current_bbox[2] / page_w, 3),
                "is_sensitive": any(x in f"{seg['content']} {valeur_associee}".upper() for x in ["@", "CLIENT", "TEL"])
            }
            seg["assigned"] = True
        else:
            if not seg["assigned"]:
                final_report[f"ZONE_{i}"] = {
                    "content": seg["content"],
                    "top": round(seg["bbox"][1] / page_h, 3),
                    "bottom": round(seg["bbox"][3] / page_h, 3),
                    "left": round(seg["bbox"][0] / page_w, 3),
                    "right": round(seg["bbox"][2] / page_w, 3),
                    "is_sensitive": "@" in seg["content"] or "CLIENT" in content_up
                }

    return final_report

# --- EXECUTION ---
try:
    report = process_hybrid_invoice("facture_FAC-2025-00001_2.pdf")
    print(json.dumps(report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")