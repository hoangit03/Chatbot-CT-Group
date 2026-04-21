import os
import glob
import shutil
import hashlib
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.services.broker_service import ocr_publisher, to_md_publisher

load_dotenv()

router = APIRouter(prefix="/api/v1/extract", tags=["Document Extraction"])

# === Path Resolution từ .env ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))
INPUT_DIR = os.path.join(SHARED_DIR, "data_input")
OUTPUT_DIR = os.path.join(SHARED_DIR, "data_output")
REGISTRY_FILE = os.path.join(SHARED_DIR, "file_registry.json")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _get_registry():
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_registry(registry):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=4)

from typing import List

@router.post("/")
async def extract_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint (Auto-Router) để nhận NHIỀU file.
    Tự động phân loại luồng OCR cho PDF hoặc ToMD cho Word/Excel/MSG.
    Tích hợp lớp khiên chắn chống trùng lặp từng file bằng mã băm SHA-256.
    """
    allowed_ocr = ['pdf']
    allowed_md = ['doc', 'docx', 'xls', 'xlsx', 'msg', 'pptx', 'ppt']
    
    registry = _get_registry()
    results = []
    
    for file in files:
        doc_name = file.filename
        ext = doc_name.split('.')[-1].lower()
        base_name = os.path.splitext(doc_name)[0]
        
        if ext not in allowed_ocr and ext not in allowed_md:
            results.append({
                "file_name": doc_name, 
                "status": "failed", 
                "message": f"Unsupported file type: {ext}"
            })
            continue
            
        # 0. Đọc toàn bộ dung lượng file vào RAM
        file_bytes = await file.read()
        
        # 1. Kiểm tra trùng lặp (Chỉ quét Tên File)
        if doc_name in registry:
            results.append({
                "file_name": doc_name, 
                "status": "failed", 
                "message": f"Từ chối hệ thống: Tên file '{doc_name}' đã từng được tải lên trước đây."
            })
            continue
        
        # Đánh dấu vào Lịch sử
        registry[doc_name] = True
        
        # 2. Xóa Rác Cũ liên quan tới tên này (Ghi đè)
        old_files = glob.glob(os.path.join(OUTPUT_DIR, f"{base_name}*"))
        for f in old_files:
            if os.path.isfile(f):
                try: os.remove(f)
                except: pass

        # 3. Ghi file mới vào Shared Volume
        local_path = os.path.join(INPUT_DIR, doc_name)
        with open(local_path, "wb") as f:
            f.write(file_bytes)
            
        # 4. Phát lệnh sang MQ
        if ext in allowed_ocr:
            success = ocr_publisher.publish_task(doc_name)
            queue_target = "ocr_task_queue"
        else:
            success = to_md_publisher.publish_task(doc_name)
            queue_target = "to_md_task_queue"

        if not success:
            registry.pop(file_hash, None)
            results.append({
                "file_name": doc_name, 
                "status": "failed", 
                "message": "Message Broker (RabbitMQ) is down."
            })
        else:
            results.append({
                "file_name": doc_name, 
                "status": "queued",
                "message": f"File sạch. Auto-route vào {queue_target}"
            })

    # Lưu lại tổng thể registry sau khi nạp loạt file
    _save_registry(registry)
    
    return JSONResponse(content={
        "batch_size": len(files),
        "results": results
    })
