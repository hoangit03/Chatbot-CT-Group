import requests
import base64
import time
import fitz

pdf_path = r"d:\CTGroup\CT_Knowledge\report.pdf"
doc = fitz.open(pdf_path)
page = doc.load_page(0)
pix = page.get_pixmap(dpi=150)
img_bytes = pix.tobytes("jpg")
b64 = base64.b64encode(img_bytes).decode("utf-8")

payload = {
    "file": b64,
}

response = requests.post("http://localhost:8080/layout-parsing", json=payload, timeout=600)
print("Status:", response.status_code)
try:
    print(response.json())
except:
    pass
