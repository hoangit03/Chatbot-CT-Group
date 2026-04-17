import os
import json
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# === Path Resolution từ .env ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))
CHROMA_DB_DIR = os.getenv("VECTOR_DB_DIR", os.path.join(REPO_ROOT, "vectorstore", "chroma_db"))

CHUNKS_DIR = os.path.join(SHARED_DIR, "data_output", "result_chunks")

os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# === Embedding Model Selection dựa trên EMBED_DEVICE ===
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu").lower()

if EMBED_DEVICE == "cuda":
    EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL_GPU", "intfloat/multilingual-e5-large")
    print(f"[Embedding Engine] Nạp model GPU: {EMBED_MODEL_NAME} trên CUDA...")
else:
    EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL_CPU", "paraphrase-multilingual-MiniLM-L12-v2")
    print(f"[Embedding Engine] Nạp model CPU: {EMBED_MODEL_NAME} trên CPU...")

embedder = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
print("[Embedding Engine] Nạp mô hình hoàn tất!")

# Khởi tạo VectorDB Client thông qua Docker
chroma_client = chromadb.HttpClient(host="localhost", port=8002)

def get_collection():
    return chroma_client.get_or_create_collection(
        name="hr_policies",
        metadata={"hnsw:space": "cosine"}
    )

def run_embedding_pipeline(json_file_name: str) -> bool:
    """Đọc JSON Chunks và lưu cất vào ChromaDB."""
    collection = get_collection()
    json_path = os.path.join(CHUNKS_DIR, json_file_name)
    if not os.path.exists(json_path):
        print(f"[Embedding Engine] Lỗi: Không tìm thấy file JSON {json_file_name}")
        return False
        
    print(f"\n[Embedding Engine] Kích hoạt tiến trình cấy ghép Vector cho: {json_file_name}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    if not chunks:
        print(f"[Embedding Engine] File JSON rỗng: {json_file_name}")
        return True  # Không lỗi, nhưng chả có gì để nhét
        
    # Chuẩn bị dữ liệu cho Chroma
    batch_ids = []
    batch_documents = []
    batch_metadatas = []
    
    # Lấy Metadata File gốc (Ví dụ lấy từ tên file)
    base_file_name = json_file_name.replace("_chunks.json", "")
    
    for chunk in chunks:
        c_id = chunk["chunk_id"]
        c_text = chunk["content"]
        c_meta = chunk["metadata"]
        
        # Bổ sung metadata cấp file
        c_meta["source_file"] = base_file_name
        
        batch_ids.append(c_id)
        batch_documents.append(c_text)
        batch_metadatas.append(c_meta)

    # Lấy dimension thực tế từ model
    embedding_dim = embedder.get_sentence_embedding_dimension()
    print(f"[Embedding Engine] Đang ép {len(chunks)} Chunks thành Vector ({embedding_dim} chiều)...")
    
    # Tạo Embeddings
    embeddings = embedder.encode(batch_documents, show_progress_bar=False).tolist()
    
    print("[Embedding Engine] Tiến hành nhét vào kho ChromaDB (Upsert)...")
    collection.upsert(
        ids=batch_ids,
        embeddings=embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas
    )
    
    print(f"[Embedding Engine] Thành công! Hệ thống RAG đã hấp thụ {json_file_name}.")
    return True
