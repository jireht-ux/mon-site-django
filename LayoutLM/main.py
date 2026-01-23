import fitz
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json

# 1. Initialisation du modèle
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_hybrid_invoice_v5(pdf_path):
    doc = fitz.open(pdf_path)
    if doc.page_count == 0: return {"error": "PDF vide"}
    
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 2. Extraction des segments physiques
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

    # 3. INTERVENTION DU MODÈLE IA
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
        # On stocke le label prédit par l'IA (HEADER, QUESTION, ANSWER, etc.)
        segments[i]["label_ia"] = model.config.id2label[p_idx].replace("B-","").replace("I-","")

    # 4. CLUSTERING ADAPTATIF (Basé sur les labels et la position)
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
                    v_dist = abs(other["bbox"][1] - member["bbox"][1])
                    v_gap = other["bbox"][1] - member["bbox"][3]
                    h_gap = max(0, other["bbox"][0] - member["bbox"][2], member["bbox"][0] - other["bbox"][2])
                    
                    # LOGIQUE A : En-tête (HEADER) -> Priorité au regroupement VERTICAL
                    if member["label_ia"] == "HEADER" or "FACTURE" in member["content"].upper():
                        # Si aligné à gauche et juste en dessous (ex: FACTURE N° + FAC-2025)
                        if abs(other["bbox"][0] - member["bbox"][0]) < 15 and v_gap < 12:
                            is_close = True
                    
                    # LOGIQUE B : Corps/Totaux (QUESTION/ANSWER) -> Priorité HORIZONTALE
                    # Si c'est sur la même ligne (ex: Total TTC + 223.66)
                    elif v_dist < 8 and h_gap < 100:
                        is_close = True
                    
                    # LOGIQUE C : Bloc Adresse (Proximité générale)
                    elif v_gap < 10 and h_gap < 50:
                        is_close = True

                    if is_close: break
                
                if is_close:
                    cluster.append(other)
                    available_segments.remove(other)
                    added = True
        macro_clusters.append(cluster)

    # 5. Formatage du résultat attendu
    final_report = {}
    for idx, cluster in enumerate(macro_clusters):
        sorted_cluster = sorted(cluster, key=lambda x: (round(x["bbox"][1]), x["bbox"][0]))
        content = " ".join([c["content"] for c in sorted_cluster])
        
        # On prend le label le plus pertinent du groupe
        labels = [c["label_ia"] for c in cluster if c["label_ia"] != "O"]
        final_label = labels[0] if labels else "ZONE"
        
        b = [min(c["bbox"][0] for c in cluster), min(c["bbox"][1] for c in cluster),
             max(c["bbox"][2] for c in cluster), max(c["bbox"][3] for c in cluster)]
        
        final_report[f"{final_label}_{idx}"] = {
            "content": content,
            "top": round(b[1] / page_h, 3),
            "bottom": round(b[3] / page_h, 3),
            "left": round(b[0] / page_w, 3),
            "right": round(b[2] / page_w, 3),
            "is_sensitive": any(x in content.upper() for x in ["@", "TEL", "CLIENT"])
        }
    return final_report

# Test
print(json.dumps(process_hybrid_invoice_v5("facture_2.pdf"), indent=4, ensure_ascii=False))