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
            raise ValueError(f"Không hỗ trợ VECTOR_STORE_TYPE = {store_type}. Hỗ trợ: {list(cls._stores.keys())}")
        
        store_class = cls._stores[store_type]
        return store_class()