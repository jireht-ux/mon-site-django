import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json
import re

# 1. Initialisation du cerveau (IA)
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_hybrid_invoice(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 2. Extraction chirurgicale des segments (La structure)
    blocks_raw = page.get_text("dict")["blocks"]
    segments = []
    for b in blocks_raw:
        if "lines" not in b: continue
        for l in b["lines"]:
            for span in l["spans"]:
                txt = span["text"].strip()
                if not txt: continue
                segments.append({
                    "bbox": span["bbox"], # (x0, y0, x1, y1)
                    "content": txt,
                    "label_ia": "O",
                    "assigned": False
                })

    # 3. Phase d'étiquetage par l'IA
    words = [s["content"] for s in segments]
    # Normalisation pour l'IA (0-1000)
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
    # On assigne les labels de l'IA à nos segments
    for i, p_idx in enumerate(predictions[:len(segments)]):
        segments[i]["label_ia"] = model.config.id2label[p_idx].replace("B-","").replace("I-","")

    # 4. Phase de recherche spatiale (Ancres IA + Géométrie)
    final_report = {}
    
    # On définit ce qui sert d'ancre (Libellés potentiels)
    labels_to_complete = ["HEADER", "QUESTION", "LABEL"] 
    # Mots clés de sécurité si l'IA rate le label
    keywords = ["TVA", "TOTAL", "TTC", "HT", "FACTURE", "DATE"]

    for i, seg in enumerate(segments):
        if seg["assigned"]: continue
        
        content_up = seg["content"].upper()
        # On déclenche la recherche si l'IA dit que c'est un label OU si mot-clé détecté
        is_anchor = seg["label_ia"] in labels_to_complete or any(k in content_up for k in keywords)

        if is_anchor:
            valeur_associee = ""
            # --- Recherche à DROITE ---
            for j in range(i + 1, len(segments)):
                cand = segments[j]
                if cand["assigned"]: continue
                
                same_line = abs(seg["bbox"][1] - cand["bbox"][1]) < 8 # Tolérance pixels
                at_right = cand["bbox"][0] > (seg["bbox"][2] - 5)
                
                if same_line and at_right:
                    valeur_associee = cand["content"]
                    cand["assigned"] = True
                    break
            
            # --- Recherche en DESSOUS (si rien à droite) ---
            if not valeur_associee:
                for j in range(i + 1, len(segments)):
                    cand = segments[j]
                    if cand["assigned"]: continue
                    
                    is_below = 0 < (cand["bbox"][1] - seg["bbox"][3]) < 20
                    is_aligned = abs(seg["bbox"][0] - cand["bbox"][0]) < 30
                    
                    if is_below and is_aligned:
                        valeur_associee = cand["content"]
                        cand["assigned"] = True
                        break

            # Construction du bloc final
            key_name = f"{seg['label_ia']}_{i}" if seg['label_ia'] != "O" else f"DATA_{i}"
            final_report[key_name] = {
                "content": f"{seg['content']} {valeur_associee}".strip(),
                "top": round(seg["bbox"][1] / page_h, 3),
                "is_sensitive": "@" in seg["content"] or "CLIENT" in content_up
            }
            seg["assigned"] = True
        else:
            # Reste des données (si pas une ancre)
            if not seg["assigned"]:
                final_report[f"ZONE_{i}"] = {
                    "content": seg["content"],
                    "top": round(seg["bbox"][1] / page_h, 3),
                    "is_sensitive": "@" in seg["content"] or "CLIENT" in content_up
                }

    return final_report

# --- EXECUTION ---
try:
    report = process_hybrid_invoice("facture_FAC-2025-00001_2.pdf")
    print(json.dumps(report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")