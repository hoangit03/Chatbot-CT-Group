import os
import glob
import shutil
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

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.post("/")
async def extract_document(file: UploadFile = File(...)):
    """
    Endpoint (Auto-Router) để nhận file.
    Tự động phân loại luồng OCR cho PDF hoặc ToMD cho Word/Excel/MSG.
    """
    ext = file.filename.split('.')[-1].lower()
    allowed_ocr = ['pdf']
    allowed_md = ['doc', 'docx', 'xls', 'xlsx', 'msg', 'pptx', 'ppt']
    
    if ext not in allowed_ocr and ext not in allowed_md:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        
    doc_name = file.filename
    base_name = os.path.splitext(doc_name)[0]
    
    # 1. Nhận file vào Shared Volume
    local_path = os.path.join(INPUT_DIR, doc_name)
    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    # 2. Xóa Rác Cũ
    old_files = glob.glob(os.path.join(OUTPUT_DIR, f"{base_name}*"))
    for f in old_files:
        if os.path.isfile(f):
            try:
                os.remove(f)
            except:
                pass

    # 3. Auto-Routing Decision
    if ext in allowed_ocr:
        success = ocr_publisher.publish_task(doc_name)
        queue_target = "ocr_task_queue"
    else:
        success = to_md_publisher.publish_task(doc_name)
        queue_target = "to_md_task_queue"

    if not success:
        raise HTTPException(status_code=500, detail="Message Broker (RabbitMQ) is down.")
        
    return JSONResponse(content={
        "status": "queued",
        "message": f"File đã được chuyển phát auto-route vào {queue_target}",
        "file_name": doc_name
    })
