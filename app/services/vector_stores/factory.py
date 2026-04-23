import os
from typing import Type

from app.services.vector_stores.base import BaseVectorStore
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)
class VectorStoreFactory:
    """Factory Pattern"""

    _stores = {
        "chroma": "app.services.vector_stores.chroma.ChromaVectorStore",
        "qdrant": "app.services.vector_stores.qdrant.QdrantVectorStore",
    }

    _resolved: dict[str, type] = {}

    @classmethod
    def get_vector_store(cls) -> BaseVectorStore:
        store_type = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()
        store_class = cls._resolve_class(store_type)
        logger.info(f"[VectorStoreFactory] Tạo vector store: '{store_type}'")
        return store_class()
    
    @classmethod
    def register(cls, name: str, store_class: type) -> None:
        """
        """
        if not issubclass(store_class, BaseVectorStore):
            raise TypeError(f"{store_class} phải kế thừa BaseVectorStore")
        cls._resolved[name.lower()] = store_class
        logger.info(f"[VectorStoreFactory] Đã đăng ký store: '{name}'")

    @classmethod
    def _resolve_class(cls, name: str) -> type:
        """Lazy import class từ dotted path string."""
        if name in cls._resolved:
            return cls._resolved[name]
 
        if name not in cls._registry:
            raise ValueError(
                f"VECTOR_STORE_TYPE='{name}' không được hỗ trợ. "
                f"Giá trị hợp lệ: {list(cls._registry.keys())}"
            )
 
        module_path, class_name = cls._registry[name].rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        store_class = getattr(module, class_name)
        cls._resolved[name] = store_class
        return store_class