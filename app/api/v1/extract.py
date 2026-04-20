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

@router.post("/")
async def extract_document(file: UploadFile = File(...)):
    """
    Endpoint (Auto-Router) để nhận file.
    Tự động phân loại luồng OCR cho PDF hoặc ToMD cho Word/Excel/MSG.
    Đã tích hợp lớp khiên chắn chống trùng lặp bằng mã băm SHA-256.
    """
    ext = file.filename.split('.')[-1].lower()
    allowed_ocr = ['pdf']
    allowed_md = ['doc', 'docx', 'xls', 'xlsx', 'msg', 'pptx', 'ppt']
    
    if ext not in allowed_ocr and ext not in allowed_md:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        
    doc_name = file.filename
    base_name = os.path.splitext(doc_name)[0]
    
    # 0. Đọc toàn bộ dung lượng file vào RAM
    file_bytes = await file.read()
    
    # 1. Thuật toán kiểm tra trùng lặp tàn khốc bằng SHA-256
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    registry = _get_registry()
    
    if file_hash in registry:
        # Nếu mã băm đã tồn tại trong RAG => Từ chối luôn lập tức (Xài HTTP 409 Conflict)
        old_file_name = registry[file_hash]
        raise HTTPException(
            status_code=409, 
            detail=f"Trùng lặp dữ liệu: Nội dung file này đã tồn tại trong Hệ thống dưới tên '{old_file_name}'. Vui lòng không upload lại để tránh rác sinh ra trong AI."
        )
    
    # Lưu danh tính file mới vào Lịch sử
    registry[file_hash] = doc_name
    _save_registry(registry)
    
    # 2. Nhận file vào Shared Volume
    local_path = os.path.join(INPUT_DIR, doc_name)
    with open(local_path, "wb") as f:
        f.write(file_bytes)
        
    # 3. Xóa Rác Cũ liên quan tới tên này (Đề phòng trường hợp ghi đè)
    old_files = glob.glob(os.path.join(OUTPUT_DIR, f"{base_name}*"))
    for f in old_files:
        if os.path.isfile(f):
            try:
                os.remove(f)
            except:
                pass

    # 4. Auto-Routing Decision
    if ext in allowed_ocr:
        success = ocr_publisher.publish_task(doc_name)
        queue_target = "ocr_task_queue"
    else:
        success = to_md_publisher.publish_task(doc_name)
        queue_target = "to_md_task_queue"

    if not success:
        # Xóa Lịch sử nếu rớt mạng MQ
        registry.pop(file_hash, None)
        _save_registry(registry)
        raise HTTPException(status_code=500, detail="Message Broker (RabbitMQ) is down.")
        
    return JSONResponse(content={
        "status": "queued",
        "message": f"File sạch (Không trùng). Đã được chuyển phát auto-route vào {queue_target}",
        "file_name": doc_name
    })
