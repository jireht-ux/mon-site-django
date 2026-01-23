import fitz
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json

# 1. Initialisation
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_hybrid_invoice_final(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 2. Extraction des segments
    segments = []
    blocks_raw = page.get_text("dict")["blocks"]
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

    # 3. Prédiction LayoutLMv3
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

    # 4. Clustering Intelligent
    macro_clusters = []
    available_segments = sorted(segments, key=lambda x: (x["bbox"][1], x["bbox"][0]))
    
    while available_segments:
        current = available_segments.pop(0)
        cluster = [current]
        
        added = True
        while added:
            added = False
            for other in list(available_segments):
                is_close = False
                for member in cluster:
                    v_dist = abs(other["bbox"][1] - member["bbox"][1])
                    v_gap = other["bbox"][1] - member["bbox"][3]
                    h_gap = max(0, other["bbox"][0] - member["bbox"][2], member["bbox"][0] - other["bbox"][2])
                    
                    content_all = (member["content"] + " " + other["content"]).upper()
                    
                    # RÈGLE 1 : Métadonnées En-tête (Date, Échéance) -> Fusion Verticale
                    if any(k in content_all for k in ["DATE", "ÉCHÉANCE", "N°"]):
                        if abs(other["bbox"][0] - member["bbox"][0]) < 20 and v_gap < 15:
                            is_close = True
                    
                    # RÈGLE 2 : Totaux (TVA, TTC, HT) -> Fusion Horizontale Large
                    elif any(k in content_all for k in ["TVA", "TOTAL", "TTC", "HT"]):
                        if v_dist < 10 and h_gap < 250: # On accepte un grand vide horizontal pour lier libellé et montant
                            is_close = True
                    
                    # RÈGLE 3 : Blocs de texte standard (Adresses, etc.)
                    elif v_gap < 12 and h_gap < 60:
                        is_close = True

                    if is_close: break
                
                if is_close:
                    cluster.append(other)
                    available_segments.remove(other)
                    added = True
        macro_clusters.append(cluster)

    # 5. Génération du rapport
    final_report = {}
    for idx, cluster in enumerate(macro_clusters):
        sorted_cluster = sorted(cluster, key=lambda x: (round(x["bbox"][1], 1), x["bbox"][0]))
        content = " ".join([c["content"] for c in sorted_cluster])
        
        b = [min(c["bbox"][0] for c in cluster), min(c["bbox"][1] for c in cluster),
             max(c["bbox"][2] for c in cluster), max(c["bbox"][3] for c in cluster)]
        
        label = cluster[0]["label_ia"] if cluster[0]["label_ia"] != "O" else "ZONE"
        
        final_report[f"{label}_{idx}"] = {
            "content": content,
            "top": round(b[1] / page_h, 3),
            "bottom": round(b[3] / page_h, 3),
            "left": round(b[0] / page_w, 3),
            "right": round(b[2] / page_w, 3),
            "is_sensitive": any(x in content.upper() for x in ["@", "TEL", "CLIENT"])
        }
    return final_report

print(json.dumps(process_hybrid_invoice_final("facture_2.pdf"), indent=4, ensure_ascii=False))