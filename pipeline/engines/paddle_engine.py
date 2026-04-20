import os
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# === Path Resolution từ .env (thống nhất toàn hệ thống) ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))

INPUT_DIR = os.path.join(SHARED_DIR, "data_input")
OUTPUT_DIR = os.path.join(SHARED_DIR, "data_output")
RESULT_DIR = os.path.join(OUTPUT_DIR, "result")

# Định tuyến qua API Endpoint Phân Tán
# Sử dụng os.getenv thay vì hardcode localhost
PADDLE_OCR_ENDPOINT = os.getenv("PADDLE_OCR_ENDPOINT", "http://localhost:8080/layout-parsing")

def run_ocr_pipeline(pdf_name: str) -> bool:
    """
    Kích hoạt tiến trình Paddle OCR qua HTTP Server API.
    Architecture: Distributed Microservices Cấp Độ Enterprise.
    Hỗ trợ xử lý trực tiếp base64 của file, PaddleX tự động gom markdown.
    """
    base_name = os.path.splitext(pdf_name)[0]
    print(f"\n[Paddle Engine HTTP] Bắn lệnh xử lý từ xa cho file: {pdf_name}", flush=True)
    
    file_path = os.path.join(INPUT_DIR, pdf_name)
    if not os.path.exists(file_path):
        print(f"[Engine Error] Không tìm thấy file gốc tại {file_path}")
        return False
        
    try:
        # Nạp tệp thành Data Byte
        with open(file_path, "rb") as f:
            file_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[Engine Error] Lỗi đọc Binary cho file {pdf_name}: {e}")
        return False

    payload = {
        "file_b64": file_b64,
        "file_name": pdf_name
    }
    
    try:
        # PING TRẠM TỪ XA
        print(f"[Paddle Engine HTTP] Đang truyền tải gói dữ liệu {len(file_b64)/1024:.2f} KB tới Máy 4GB OCR...", flush=True)
        res = requests.post(PADDLE_OCR_ENDPOINT, json=payload, timeout=600)
        res.raise_for_status()
        
        api_response = res.json()
        if not api_response.get("success", False):
            print(f"[Engine Error] PaddleX Từ Chối với mã lỗi: {api_response.get('error')}")
            return False
            
    except Exception as e:
        print(f"[Engine Error] Không gửi được tới {PADDLE_OCR_ENDPOINT} - Exception: {e}")
        return False
        
    # Xử lý Kết Quả Trả Về (Native Markdown của API nội bộ trả rắng)
    try:
        markdown_result = api_response.get("markdown", "")
        if not markdown_result:
            print("[Engine Error] Nhận được kết quả rỗng!")
            return False
        
        os.makedirs(RESULT_DIR, exist_ok=True)
        final_md_path = os.path.join(RESULT_DIR, f"{base_name}_paddle_only.md")
        
        with open(final_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_result)
            
        print(f"[Paddle Engine HTTP] Đã ghim thành công cấu trúc Markdown RAG cho {pdf_name}")
        return True
        
    except Exception as e:
        print(f"[Engine Error] Sập Parser Kết Quả JSON: {e}")
        return False
