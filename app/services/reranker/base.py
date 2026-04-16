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