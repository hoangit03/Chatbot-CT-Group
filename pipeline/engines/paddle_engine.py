import os
import json
import base64
import time
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

# Retry config cho external API (ngrok, cloud, etc.)
ENGINE_MAX_RETRIES = 5
ENGINE_BASE_DELAY = 30  # seconds

# Header bypass ngrok free tier interstitial page
NGROK_HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "CTGroup-OCR-Engine/1.0"
}

def run_ocr_pipeline(pdf_name: str) -> bool:
    """
    Kích hoạt tiến trình Paddle OCR qua HTTP Server API.
    Architecture: Distributed Microservices Cấp Độ Enterprise.
    Hỗ trợ xử lý trực tiếp base64 của file, PaddleX tự động gom markdown.
    Tích hợp retry + ngrok bypass header.
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
    
    # === Retry loop với exponential backoff ===
    api_response = None
    for attempt in range(1, ENGINE_MAX_RETRIES + 1):
        try:
            print(f"[Paddle Engine HTTP] Lần {attempt}/{ENGINE_MAX_RETRIES} — Truyền tải {len(file_b64)/1024:.2f} KB...", flush=True)
            res = requests.post(
                PADDLE_OCR_ENDPOINT, 
                json=payload, 
                headers=NGROK_HEADERS,
                timeout=7200
            )
            
            # Ngrok trả HTML interstitial thay vì JSON -> retry
            content_type = res.headers.get("Content-Type", "")
            if "text/html" in content_type:
                print(f"[Engine Warn] Nhận HTML thay vì JSON (ngrok interstitial?) — thử lại...", flush=True)
                if attempt < ENGINE_MAX_RETRIES:
                    delay = ENGINE_BASE_DELAY * attempt
                    print(f"[Engine] Đợi {delay}s...", flush=True)
                    time.sleep(delay)
                continue
            
            # 503/502/429 -> server bận hoặc ngrok rate limit -> retry
            if res.status_code in (502, 503, 429):
                print(f"[Engine Warn] HTTP {res.status_code} — server bận hoặc ngrok rate limit", flush=True)
                if attempt < ENGINE_MAX_RETRIES:
                    delay = ENGINE_BASE_DELAY * attempt
                    print(f"[Engine] Đợi {delay}s trước khi thử lại...", flush=True)
                    time.sleep(delay)
                continue
            
            res.raise_for_status()
            api_response = res.json()
            
            if not api_response.get("success", False):
                print(f"[Engine Error] PaddleX Từ Chối: {api_response.get('error', 'unknown')}")
                return False
            
            # Thành công -> thoát loop
            break
            
        except requests.exceptions.ConnectionError as e:
            print(f"[Engine Warn] Connection Error lần {attempt}: {e}", flush=True)
            if attempt < ENGINE_MAX_RETRIES:
                delay = ENGINE_BASE_DELAY * attempt
                print(f"[Engine] Đợi {delay}s...", flush=True)
                time.sleep(delay)
            continue
            
        except requests.exceptions.Timeout:
            print(f"[Engine Warn] Timeout lần {attempt} (7200s)", flush=True)
            if attempt < ENGINE_MAX_RETRIES:
                time.sleep(ENGINE_BASE_DELAY)
            continue
            
        except Exception as e:
            print(f"[Engine Error] Lỗi không xác định lần {attempt}: {e}", flush=True)
            if attempt < ENGINE_MAX_RETRIES:
                time.sleep(ENGINE_BASE_DELAY)
            continue
    
    if api_response is None:
        print(f"[Engine Error] Thất bại sau {ENGINE_MAX_RETRIES} lần thử cho {pdf_name}")
        return False
        
    # Xử lý Kết Quả Trả Về
    try:
        markdown_result = api_response.get("markdown", "")
        if not markdown_result:
            print("[Engine Error] Nhận được kết quả rỗng!")
            return False
        
        os.makedirs(RESULT_DIR, exist_ok=True)
        final_md_path = os.path.join(RESULT_DIR, f"{base_name}_paddle_only.md")
        
        with open(final_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_result)
            
        print(f"[Paddle Engine HTTP] ✅ Đã ghim thành công {pdf_name} ({len(markdown_result)} chars)", flush=True)
        return True
        
    except Exception as e:
        print(f"[Engine Error] Sập Parser Kết Quả JSON: {e}")
        return False
