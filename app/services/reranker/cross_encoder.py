from typing import List, Tuple
import os
import torch
from dotenv import load_dotenv

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.services.reranker.base import BaseReranker

load_dotenv()

class CrossEncoderReranker(BaseReranker):
    """Concrete implementation dùng Cross-Encoder (rất hiệu quả cho RAG)"""

    def __init__(self, model_name: str = None):
        self.device = os.getenv("DEVICE","cpu")
        self.model_name = model_name or "Qwen/Qwen3-Reranker-0.6B"
        print(f"Khởi tạo CrossEncoder reranker: {self.model_name} xử lý GPU: {self.device}, cuda: {torch.cuda.is_available()}")
        self.model = CrossEncoder(self.model_name, device=self.device)  

    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        if not documents:
            return []

        # Tạo pairs (query, doc_content)
        pairs = [[query, doc.page_content] for doc in documents]

        # Predict scores
        scores = self.model.predict(pairs)

        # Kết hợp document + score
        scored_docs = list(zip(documents, scores))

        # Sắp xếp theo score giảm dần
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs