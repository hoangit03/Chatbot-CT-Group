import os
import time
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Embedder API", version="1.0", description="OpenAI-compatible Embedding Service")

model_name = os.getenv("EMBEDDING_MODEL_GPU", "intfloat/multilingual-e5-large")
device = os.getenv("EMBED_DEVICE", "cuda").lower()

def robust_download(repo_id: str):
    """Stubborn download function that retries indefinitely on SSL/Network errors"""
    max_retries = 100
    for attempt in range(max_retries):
        try:
            print(f"[Embedder API] Đang kiểm tra/tải mô hình {repo_id} (Lần thử {attempt + 1}/{max_retries})...")
            # resume_download=True cho phép tải tiếp từ chỗ đứt gãy
            snapshot_download(repo_id, resume_download=True)
            print(f"[Embedder API] Hoàn tất tải mô hình {repo_id}!")
            return
        except Exception as e:
            print(f"[Embedder API] Lỗi tải mô hình (thường do mạng): {e}")
            print(f"[Embedder API] Sẽ thử lại sau 5 giây...")
            time.sleep(5)
            
# 1. Tải mô hình lì lợm
robust_download(model_name)

# 2. Nạp mô hình vào RAM/VRAM
print(f"[Embedder API] Đang nạp mô hình {model_name} lên {device.upper()}...")
model = SentenceTransformer(model_name, device=device)
print(f"[Embedder API] Sẵn sàng phục vụ!")

# 3. Định nghĩa API chuẩn OpenAI
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = model_name

@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    # Đảm bảo input luôn là list
    inputs = req.input if isinstance(req.input, list) else [req.input]
    
    # Tính toán Vector
    embeddings = model.encode(inputs, normalize_embeddings=True).tolist()
    
    # Định dạng chuẩn OpenAI
    data = []
    for i, emb in enumerate(embeddings):
        data.append({
            "object": "embedding",
            "embedding": emb,
            "index": i
        })
        
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }
