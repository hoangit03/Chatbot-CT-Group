from fastapi import APIRouter
from pipeline.engines.vectordb_engine import get_all_vectors, delete_all_vectors

router = APIRouter(prefix="/api/v1/vectordb", tags=["VectorDB Analytics"])

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
    return result
