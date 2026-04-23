import requests
import base64
import time
import os

def test():
    pdf_path = r"d:\CTGroup\CT_Knowledge\report.pdf"
    if not os.path.exists(pdf_path):
        print("File not found")
        return
        
    with open(pdf_path, "rb") as f:
        file_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {
        "file_name": "report.pdf",
        "file_b64": file_b64
    }
    
    print("Sending Base64 to Custom FastAPI Server...")
    start = time.time()
    try:
        response = requests.post("http://localhost:8080/layout-parsing", json=payload, timeout=600)
        print("Status Code:", response.status_code)
        print("Time:", time.time() - start)
        print("Response JSON:")
        try:
            data = response.json()
            print("Success:", data.get("success"))
            print("Markdown Length:", len(data.get("markdown", "")))
            if data.get("error"):
                print("Error:", data.get("error"))
        except:
            print("Raw text:", response.text[:500])
    except Exception as e:
        print("Failed request:", e)

if __name__ == "__main__":
    test()
