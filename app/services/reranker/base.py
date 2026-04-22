from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain_core.documents import Document


class BaseReranker(ABC):
    """Abstract Base Class cho Reranker - dễ thay mô hình khác sau này"""

    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Trả về list (document, rerank_score) đã sắp xếp theo độ liên quan giảm dần
        """
        pass

    async def arerank(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Bất đồng bộ — mặc định offload rerank() sang asyncio.to_thread().
        """
        import asyncio
        return await asyncio.to_thread(self.rerank, query, documents)