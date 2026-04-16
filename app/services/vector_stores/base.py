from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class BaseVectorStore(ABC):
    """Abstract Base Class theo SOLID - Dễ mở rộng bất kỳ Vector DB nào"""

    @abstractmethod
    def add_documents(self, documents: List[Document], embedding: Embeddings, replace: bool) -> None:
        """Thêm documents vào vector store (tạo mới hoặc replace)"""
        pass

    @abstractmethod
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Trả về retriever để dùng trong RAG"""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Xóa toàn bộ collection (dùng khi re-ingest)"""
        pass