import os
from fastapi import APIRouter
from pipeline.engines.vectordb_engine import get_all_vectors, delete_all_vectors
from app.services.cache_service import SemanticCache
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))
REGISTRY_FILE = os.path.join(SHARED_DIR, "file_registry.json")

router = APIRouter(prefix="/etl/v1/vectordb", tags=["VectorDB Analytics"])

@router.get("/all")
async def fetch_all_vectordb_data():
    """
    Endpoint trả về toàn bộ dữ liệu chứa trong VectorDB ChromaDB (Bao gồm ID, Metadata, Text và Embeddings)
    """
    result = get_all_vectors()
    return result

@router.delete("/all")
async def clear_all_vectordb_data():
    """
    Endpoint (Danger Zone): Bấm phát xóa trắng toàn bộ VectorDB để test Data mới từ đầu.
    """
    result = delete_all_vectors()
    
    # Xóa file tracking để cho phép upload lại file cũ
    if os.path.exists(REGISTRY_FILE):
        try:
            os.remove(REGISTRY_FILE)
            result["message"] += " Đã xóa lịch sử nạp file (file_registry.json)."
        except Exception as e:
            result["message"] += f" Lỗi xóa lịch sử nạp file: {e}"
            
    # Flush Semantic Cache
    SemanticCache().flush_cache()
    
    return result
