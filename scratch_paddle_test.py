import requests
import base64
import time
import os

def test_paddleserve_pdf(pdf_path):
    print(f"Testing PDF via PyMuPDF: {pdf_path}")
    import fitz
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # Trang 1
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    # Payload format for openapi /layout-parsing
    # 0 đại diện cho Image (1 là PDF nhưng có vẻ dễ sập)
    payload = {
        "file": b64,
        "fileType": 0
    }
    
    print("Gửi Request Hình Ảnh tới http://localhost:8080/layout-parsing ...")
    start = time.time()
    try:
        response = requests.post("http://localhost:8080/layout-parsing", json=payload, timeout=600)
        end = time.time()
        print(f"Status Code: {response.status_code}")
        print(f"Thời gian xử lý: {end - start:.2f} giây")
        
        try:
            res_json = response.json()
            if "result" in res_json and "layoutParsingResults" in res_json["result"]:
                print("Chứa layoutParsingResults!")
                for idx, page in enumerate(res_json["result"]["layoutParsingResults"]):
                    print(f"--- Kết Quả ---")
                    print(page["markdown"].get("text", "Không có text")[:2000] + "\n")
            else:
               print("Lỗi JSON không có trường mong đợi:", str(res_json)[:1000])
        except Exception as e:
            print("Không parse được JSON:", e)
            print("Raw text:", response.text[:1000])
    except Exception as e:
        print("Lỗi Request:", e)

if __name__ == "__main__":
    test_pdf = r"d:\CTGroup\CT_Knowledge\report.pdf"
    if not os.path.exists(test_pdf):
        print("Không tìm thấy file!")
    else:
        test_paddleserve_pdf(test_pdf)
