import os
import httpx
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document

from app.services.reranker.base import BaseReranker

load_dotenv()


class CrossEncoderReranker(BaseReranker):
    """Reranker gọi Core Reranker API (GPU microservice) thay vì load model local"""

    def __init__(self, model_name: str = None):
        self.reranker_url = os.getenv("CORE_RERANKER_URL", "http://core_reranker:8009")
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
        print(f"[Reranker] Kết nối Core Reranker API tại: {self.reranker_url}")

    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        if not documents:
            return []

        # Gửi request tới Core Reranker service
        payload = {
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_k": len(documents)  # Lấy hết scores, lọc ở RetrievalService
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(f"{self.reranker_url}/rerank", json=payload)
                response.raise_for_status()
                results = response.json()
        except Exception as e:
            print(f"[Reranker] ⚠️ Lỗi gọi Core Reranker API: {e}")
            # Fallback: trả documents gốc với score = 0
            return [(doc, 0.0) for doc in documents]

        # Map scores từ API response về (Document, score)
        scored_docs = []
        for result in results:
            idx = result["index"]
            score = result["score"]
            if idx < len(documents):
                scored_docs.append((documents[idx], float(score)))

        # Sắp xếp theo score giảm dần
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs