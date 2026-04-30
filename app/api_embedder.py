"""
Embedder API — Proxy tới Core Embedding GPU Microservice
Không load model local, chỉ forward requests tới core_embedding:8004
"""
import os
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI(title="Embedder API", version="2.0", description="Proxy to Core Embedding Service")

CORE_EMBED_URL = os.getenv("CORE_EMBED_URL", "http://core_embedding:8004")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "intfloat/multilingual-e5-large"


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    """OpenAI-compatible endpoint, proxy tới Core Embedding service"""
    inputs = req.input if isinstance(req.input, list) else [req.input]

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{CORE_EMBED_URL}/embed",
            json={"inputs": inputs}
        )
        response.raise_for_status()
        result = response.json()

    # Format chuẩn OpenAI
    data = []
    for i, emb in enumerate(result["embeddings"]):
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


@app.get("/health")
async def health():
    """Kiểm tra kết nối tới Core Embedding"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CORE_EMBED_URL}/health")
            return {"status": "ok", "core_embedding": resp.json()}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
