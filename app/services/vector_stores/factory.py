import os
from typing import Type

from app.services.vector_stores.base import BaseVectorStore
from app.services.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

load_dotenv()

class VectorStoreFactory:
    """Factory Pattern"""

    _stores = {
        "chroma": ChromaVectorStore,
    }

    @classmethod
    def get_vector_store(cls) -> BaseVectorStore:
        store_type = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()
        if store_type not in cls._stores:
            raise ValueError(
                f"VECTOR_STORE_TYPE='{store_type}' không được hỗ trợ. "
                f"Các giá trị hợp lệ: {list(cls._stores.keys())}"
            )
        return cls._stores[store_type]()
    
    @classmethod
    def register(cls, name: str, store_class: type) -> None:
        """Đăng ký vector store mới từ bên ngoài — không cần sửa factory."""
        cls._stores[name.lower()] = store_class