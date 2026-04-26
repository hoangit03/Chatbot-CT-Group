import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

# === Path Resolution từ .env ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CHROMA_DB_DIR = os.getenv("VECTOR_DB_DIR", os.path.join(REPO_ROOT, "vectorstore", "chroma_db"))

# Lazy init
_chroma_client = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", 8002))
        _chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    return _chroma_client

def get_all_vectors(collection_name: str = "general"):
    """
    Lấy toàn bộ dữ liệu (trừ chuỗi Vector float quá nặng) từ ChromaDB
    """
    try:
        chroma_client = get_chroma_client()
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        return {"status": "error", "message": f"Chưa có Collection {collection_name} hoặc DB trống. Chi tiết lỗi: {str(e)}"}
        
    count = collection.count()
    if count == 0:
        return {
            "status": "success", 
            "count": 0, 
            "data": []
        }
        
    # Đã bổ sung "embeddings" vào danh sách include theo yêu cầu của Giám Đốc!
    all_data = collection.get(include=["metadatas", "documents", "embeddings"])
    
    embeddings_list = all_data.get("embeddings")
    
    formatted_data = []
    # collection.get() trả về một dictionary các list, mình phải loop array để ghép lại
    for i in range(len(all_data["ids"])):
        emb_val = []
        if embeddings_list is not None and len(embeddings_list) > i:
            raw_emb = embeddings_list[i]
            # Chống Crash Numpy sang JSON và chống lỗi ValueError (Truth value ambiguous)
            if hasattr(raw_emb, "tolist"):
                emb_val = raw_emb.tolist()
            else:
                emb_val = list(raw_emb)

        formatted_data.append({
            "id": all_data["ids"][i],
            "metadata": all_data["metadatas"][i] if all_data["metadatas"] else {},
            "content": all_data["documents"][i] if all_data["documents"] else "",
            "embedding_tensor": emb_val
        })
        
    return {
        "status": "success",
        "collection": collection_name,
        "count": count,
        "data": formatted_data
    }

def delete_all_vectors(collection_name: str = "hr_policies"):
    """
    Xoá toàn bộ dữ liệu trong Collection của ChromaDB để làm mới (Reset) DB.
    """
    try:
        # Thay vì delete_collection gây lỗi giật sập Index của SQLite, 
        # ta sẽ lấy toàn bộ ID và xóa sạch tệp nhỏ bên trong.
        chroma_client = get_chroma_client()
        collection = chroma_client.get_or_create_collection(name=collection_name)
        all_data = collection.get(include=[])
        ids = all_data.get("ids", [])
        
        if not ids:
            return {"status": "success", "message": f"Collection '{collection_name}' đã trống sẵn, không cần wipe."}
            
        # Chia lô xóa nếu quá nhiều (tuy Local RAG thường ít data nên xóa 1 lần cũng được)
        collection.delete(ids=ids)
        
        return {"status": "success", "message": f"Đã Wipe thành công {len(ids)} dữ liệu Vector trong '{collection_name}'."}
    except Exception as e:
        return {"status": "error", "message": f"Lỗi khi xóa Collection '{collection_name}': {str(e)}"}

def delete_vectors_by_source(source_file: str, collection_name: str = "hr_policies"):
    """
    Xóa toàn bộ Vector có metadata 'source_file' khớp với giá trị truyền vào.
    Dùng khi người dùng chọn Ghi đè file.
    """
    try:
        chroma_client = get_chroma_client()
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Truy vấn tìm các ID có source_file tương ứng
        all_data = collection.get(where={"source_file": source_file}, include=[])
        ids_to_delete = all_data.get("ids", [])
        
        if not ids_to_delete:
            return {"status": "success", "message": f"Không tìm thấy dữ liệu cũ của file '{source_file}' trong VectorDB để xóa."}
            
        collection.delete(ids=ids_to_delete)
        return {"status": "success", "message": f"Đã xóa thành công {len(ids_to_delete)} chunk của file '{source_file}' khỏi VectorDB."}
    except Exception as e:
        return {"status": "error", "message": f"Lỗi khi xóa vector theo source_file '{source_file}': {str(e)}"}

