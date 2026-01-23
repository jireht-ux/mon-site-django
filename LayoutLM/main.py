import fitz
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json

# 1. Initialisation
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_hybrid_invoice_v5(pdf_path):
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

    # 3. Prédiction IA (inchangé)
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

    # 4. Clustering Intelligent avec contrainte de distance horizontale
    macro_clusters = []
    # Tri par lecture : haut vers bas, puis gauche vers droite
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
                    # Calcul des distances relatives
                    v_dist = abs(other["bbox"][1] - member["bbox"][1]) # Distance entre les hauts
                    # Espace horizontal vide entre les deux segments
                    h_gap = max(0, other["bbox"][0] - member["bbox"][2], member["bbox"][0] - other["bbox"][2])
                    
                    # CONDITION DE FUSION :
                    # 1. Soit ils sont l'un au-dessus de l'autre (v_dist < 15) et alignés horizontalement
                    is_vertical_block = (0 < (other["bbox"][1] - member["bbox"][3]) < 15) and (abs(other["bbox"][0] - member["bbox"][0]) < 50)
                    
                    # 2. Soit ils sont sur la même ligne (v_dist < 8) MAIS proches (h_gap < 50 pixels)
                    is_horizontal_block = (v_dist < 8) and (h_gap < 50)
                    
                    if is_vertical_block or is_horizontal_block:
                        is_close = True
                        break
                
                if is_close:
                    cluster.append(other)
                    available_segments.remove(other)
                    added = True
        
        macro_clusters.append(cluster)

    # 5. Formatage du rapport final
    final_report = {}
    for idx, cluster in enumerate(macro_clusters):
        # Tri interne pour le texte
        sorted_cluster = sorted(cluster, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        content = " ".join([c["content"] for c in sorted_cluster])
        
        b_min_x = min(c["bbox"][0] for c in sorted_cluster)
        b_min_y = min(c["bbox"][1] for c in sorted_cluster)
        b_max_x = max(c["bbox"][2] for c in sorted_cluster)
        b_max_y = max(c["bbox"][3] for c in sorted_cluster)
        
        label = cluster[0]["label_ia"] if cluster[0]["label_ia"] != "O" else "ZONE"
        
        final_report[f"{label}_{idx}"] = {
            "content": content,
            "top": round(b_min_y / page_h, 3),
            "bottom": round(b_max_y / page_h, 3),
            "left": round(b_min_x / page_w, 3),
            "right": round(b_max_x / page_w, 3),
            "is_sensitive": any(x in content.upper() for x in ["@", "CLIENT", "TEL"])
        }

    return final_report

# TEST
print(json.dumps(process_hybrid_invoice_v5("facture_2.pdf"), indent=4, ensure_ascii=False))