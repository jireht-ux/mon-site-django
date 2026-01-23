import fitz  # PyMuPDF
import json
import re

def process_invoice_fixed(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w, page_h = page.rect.width, page.rect.height
    
    # On utilise "dict" pour avoir la structure par blocs natifs du PDF
    # Cela sépare naturellement les paragraphes et les lignes
    blocks_raw = page.get_text("dict")["blocks"]
    
    dcp_report = {}
    idx = 0
    
    patterns = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "SIRET": r'\d{14}'
    }

    for b in blocks_raw:
        if "lines" not in b: continue  # Ignore les images
        
        for l in b["lines"]:
            # On extrait le texte de la ligne
            line_text = "".join([span["text"] for span in l["spans"]]).strip()
            if not line_text: continue
            
            # Coordonnées de la ligne
            # l["bbox"] = (x0, y0, x1, y1)
            x0, y0, x1, y1 = l["bbox"]
            
            # Identification du label
            label = "ZONE"
            is_sensitive = False
            up_text = line_text.upper()
            
            if any(p in up_text for p in ["SCT", "SARL", "SAS"]): label = "EMETTEUR"
            if "FACTURE N°" in up_text: label = "HEADER"
            if "CLIENT" in up_text or "@" in up_text: is_sensitive = True
            
            for d_name, p in patterns.items():
                if re.search(p, line_text):
                    label, is_sensitive = f"ZONE_{d_name}", True

            # Ajout au dictionnaire final si pertinent
            # On utilise le Y de la ligne pour le 'top'
            dcp_report[f"{label}_{idx}"] = {
                "top": round(y0 / page_h, 3),
                "bottom": round(y1 / page_h, 3),
                "left": round(x0 / page_w, 3),
                "right": round(x1 / page_w, 3),
                "content": line_text,
                "is_sensitive": is_sensitive
            }
            idx += 1

    return dcp_report

# --- TEST ---
try:
    report = process_invoice_fixed("facture_FAC-2025-00001_2.pdf")
    print(json.dumps(report, indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")