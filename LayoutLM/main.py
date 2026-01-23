import fitz
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_hybrid_invoice_v4(pdf_path):
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        return {"error": "Document vide"}
        
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
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

    # --- STRATÉGIE DE REGROUPEMENT AJUSTÉE ---
    macro_clusters = []
    available_segments = sorted(segments, key=lambda x: (x["bbox"][1], x["bbox"][0]))
    
    while available_segments:
        current = available_segments.pop(0)
        cluster = [current]
        current["assigned"] = True
        
        added = True
        while added:
            added = False
            for other in list(available_segments):
                is_close = False
                for member in cluster:
                    # Distance verticale
                    v_dist = abs(other["bbox"][1] - member["bbox"][1])
                    v_gap = other["bbox"][1] - member["bbox"][3]
                    # Distance horizontale (espace vide entre les deux blocks)
                    h_gap = max(0, other["bbox"][0] - member["bbox"][2], member["bbox"][0] - other["bbox"][2])
                    
                    # CONDITION 1: Même ligne MAIS proche horizontalement (< 50px)
                    # Cela sépare SCT.TEST-TEST de FACTURE N°
                    same_line_close = (v_dist < 8) and (h_gap < 50)
                    
                    # CONDITION 2: Juste en dessous et aligné (Bloc Facture N° + Numéro)
                    # v_gap < 15 permet de souder FACTURE N° avec FAC-2025...
                    just_below_aligned = (0 < v_gap < 15) and (abs(other["bbox"][0] - member["bbox"][0]) < 50)
                    
                    if same_line_close or just_below_aligned:
                        is_close = True
                        break
                
                if is_close:
                    cluster.append(other)
                    available_segments.remove(other)
                    other["assigned"] = True
                    added = True
        
        macro_clusters.append(cluster)

    final_report = {}
    for idx, cluster in enumerate(macro_clusters):
        sorted_cluster = sorted(cluster, key=lambda x: (round(x["bbox"][1]), x["bbox"][0]))
        content = " ".join([c["content"] for c in sorted_cluster])
        
        min_x = min(c["bbox"][0] for c in sorted_cluster)
        min_y = min(c["bbox"][1] for c in sorted_cluster)
        max_x = max(c["bbox"][2] for c in sorted_cluster)
        max_y = max(c["bbox"][3] for c in sorted_cluster)
        
        label = sorted_cluster[0]["label_ia"] if sorted_cluster[0]["label_ia"] != "O" else "ZONE"
        
        final_report[f"{label}_{idx}"] = {
            "content": content,
            "top": round(min_y / page_h, 3),
            "bottom": round(max_y / page_h, 3),
            "left": round(min_x / page_w, 3),
            "right": round(max_x / page_w, 3),
            "is_sensitive": any(x in content.upper() for x in ["@", "CLIENT", "TEL"])
        }

    return final_report

print(json.dumps(process_hybrid_invoice_v4("facture_2.pdf"), indent=4, ensure_ascii=False))