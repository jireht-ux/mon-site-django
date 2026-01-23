import fitz
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import json

# 1. Initialisation
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

def process_hybrid_invoice_v4(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 2. Extraction
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

    # 3. IA
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

    # --- NOUVELLE STRATÉGIE DE REGROUPEMENT ---

    # 4. Étape A : On lie d'abord Clé-Valeur SANS les retirer de la liste globale
    # On crée juste des liens logiques pour ne pas perdre la donnée
    keywords = ["TVA", "TOTAL", "TTC", "HT", "FACTURE", "DATE", "ÉCHÉANCE", "N°", "SIRET"]
    
    # 5. Étape B : Clustering Spatial pur (respecte l'ordre de lecture)
    macro_clusters = []
    available_segments = sorted(segments, key=lambda x: (x["bbox"][1], x["bbox"][0])) # Tri lecture naturelle
    
    while available_segments:
        current = available_segments.pop(0)
        cluster = [current]
        current["assigned"] = True
        
        added = True
        while added:
            added = False
            for other in list(available_segments):
                # On cherche si 'other' est proche de N'IMPORTE QUEL élément déjà dans le cluster
                is_close = False
                for member in cluster:
                    v_dist = abs(other["bbox"][1] - member["bbox"][3]) # Distance verticale
                    h_overlap = not (other["bbox"][2] < member["bbox"][0] or other["bbox"][0] > member["bbox"][2])
                    
                    # Si c'est sur la même ligne ou juste en dessous
                    if abs(other["bbox"][1] - member["bbox"][1]) < 10 or (0 < other["bbox"][1] - member["bbox"][3] < 15):
                        is_close = True
                        break
                
                if is_close:
                    cluster.append(other)
                    available_segments.remove(other)
                    other["assigned"] = True
                    added = True
        
        macro_clusters.append(cluster)

    # 6. Génération du Rapport avec TRI INTERNE
    final_report = {}
    for idx, cluster in enumerate(macro_clusters):
        # IMPORTANT : On trie les éléments du cluster par Y (top) puis X (left)
        # Cela garantit que "ALI SARL" (top: 0.076, left: 0.10) passe avant "FACTURE N°" (top: 0.076, left: 0.58)
        sorted_cluster = sorted(cluster, key=lambda x: (round(x["bbox"][1]), x["bbox"][0]))
        
        content = " ".join([c["content"] for c in sorted_cluster])
        
        min_x = min(c["bbox"][0] for c in sorted_cluster)
        min_y = min(c["bbox"][1] for c in sorted_cluster)
        max_x = max(c["bbox"][2] for c in sorted_cluster)
        max_y = max(c["bbox"][3] for c in sorted_cluster)
        
        label = cluster[0]["label_ia"] if cluster[0]["label_ia"] != "O" else "ZONE"
        
        final_report[f"{label}_{idx}"] = {
            "content": content,
            "top": round(min_y / page_h, 3),
            "bottom": round(max_y / page_h, 3),
            "left": round(min_x / page_w, 3),
            "right": round(max_x / page_w, 3),
            "is_sensitive": any(x in content.upper() for x in ["@", "CLIENT", "TEL"])
        }

    return final_report

# Exécution
print(json.dumps(process_hybrid_invoice_v4("facture_8_2.pdf"), indent=4, ensure_ascii=False))