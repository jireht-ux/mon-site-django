import fitz  # PyMuPDF
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json

# 1. Initialisation
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_hybrid_invoice_with_clustering(pdf_path):
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

    # 3. Labelisation IA
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

    # 4. Étape A : Rapprochement ANCRE + VALEUR (Priorité absolue)
    structured_data = []
    keywords = ["TVA", "TOTAL", "TTC", "HT", "FACTURE", "DATE", "ÉCHÉANCE", "N°", "SIRET"]

    for i, seg in enumerate(segments):
        if seg["assigned"]: continue
        
        is_anchor = seg["label_ia"] in ["HEADER", "QUESTION"] or any(k in seg["content"].upper() for k in keywords)

        if is_anchor:
            valeur = ""
            current_bbox = list(seg["bbox"])
            # Recherche horizontale puis verticale
            for j in range(i + 1, len(segments)):
                cand = segments[j]
                if cand["assigned"]: continue
                
                same_line = abs(seg["bbox"][1] - cand["bbox"][1]) < 8
                is_below = 0 < (cand["bbox"][1] - seg["bbox"][3]) < 20 and abs(seg["bbox"][0] - cand["bbox"][0]) < 50
                
                if same_line or is_below:
                    valeur = cand["content"]
                    current_bbox[2] = max(current_bbox[2], cand["bbox"][2])
                    current_bbox[3] = max(current_bbox[3], cand["bbox"][3])
                    cand["assigned"] = True
                    break
            
            structured_data.append({
                "content": f"{seg['content']} {valeur}".strip(),
                "bbox": current_bbox,
                "type": f"MATCHED_{seg['label_ia']}"
            })
            seg["assigned"] = True

    # 5. Étape B : Regroupement des zones orphelines en "Micro-Macro-Zones"
    # On récupère tout ce qui n'a pas été assigné
    orphans = [s for s in segments if not s["assigned"]]
    
    def cluster_zones(zones, threshold_y=15):
        if not zones: return []
        # Tri par position verticale
        zones.sort(key=lambda x: x["bbox"][1])
        clusters = []
        curr_cluster = [zones[0]]
        
        for i in range(1, len(zones)):
            prev = curr_cluster[-1]
            curr = zones[i]
            
            # Si l'écart vertical est faible, on considère que c'est le même bloc (ex: adresse)
            if (curr["bbox"][1] - prev["bbox"][3]) < threshold_y:
                curr_cluster.append(curr)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [curr]
        clusters.append(curr_cluster)
        return clusters

    micro_macro_zones = cluster_zones(orphans)
    
    for cluster in micro_macro_zones:
        content = " ".join([z["content"] for z in cluster])
        bbox = [min(z["bbox"][0] for z in cluster),
                min(z["bbox"][1] for z in cluster),
                max(z["bbox"][2] for z in cluster),
                max(z["bbox"][3] for z in cluster)]
        structured_data.append({
            "content": content,
            "bbox": bbox,
            "type": "MICRO_MACRO"
        })

    # 6. Formatage du rapport final
    final_report = {}
    for i, item in enumerate(structured_data):
        b = item["bbox"]
        final_report[f"{item['type']}_{i}"] = {
            "content": item["content"],
            "top": round(b[1] / page_h, 3),
            "bottom": round(b[3] / page_h, 3),
            "left": round(b[0] / page_w, 3),
            "right": round(b[2] / page_w, 3),
            "is_sensitive": any(x in item["content"].upper() for x in ["@", "CLIENT", "TEL"])
        }

    return final_report

# --- TEST ---
try:
    report = process_hybrid_invoice_with_clustering("facture_FAC-2025-00001_2.pdf")
    print(json.dumps(report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")