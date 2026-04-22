from typing import List, Tuple, Optional
import os
import torch
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import asyncio

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


from app.services.reranker.base import BaseReranker

load_dotenv()

_RERANKER_WORKERS = int(os.getenv("RERANKER_WORKERS", 1))
_reranker_executor = ThreadPoolExecutor(
    max_workers=_RERANKER_WORKERS,
    thread_name_prefix="reranker-worker",
)

_MAX_CONCURRENT_RERANK = _RERANKER_WORKERS
_reranker_sem: Optional[asyncio.Semaphore] = None

def _get_sem() -> asyncio.Semaphore:
    """Lazy-init trong running event loop — an toàn với uvicorn --reload."""
    global _reranker_sem
    if _reranker_sem is None:
        _reranker_sem = asyncio.Semaphore(_MAX_CONCURRENT_RERANK)
    return _reranker_sem

class CrossEncoderReranker(BaseReranker):
    """Concrete implementation dùng Cross-Encoder (rất hiệu quả cho RAG)"""

    def __init__(self, model_name: str = None):
        self.device = os.getenv("EMBED_DEVICE","cpu")
        self.model_name = model_name or "Qwen/Qwen3-Reranker-0.6B"
        print(f"Khởi tạo CrossEncoder reranker: {self.model_name} xử lý GPU: {self.device}, cuda: {torch.cuda.is_available()}")
        self.model = CrossEncoder(self.model_name, device=self.device)  

    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = list(zip(documents, scores.tolist() if hasattr(scores, "tolist") else scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
    
    async def arerank(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        if not documents:
            return []
 
        loop = asyncio.get_running_loop()
        async with _get_sem():
            return await loop.run_in_executor(
                _reranker_executor,
                self.rerank,
                query,
                documents,
            )