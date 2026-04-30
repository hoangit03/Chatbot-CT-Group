import os
import httpx
from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

load_dotenv()


class CoreEmbeddings(Embeddings):
    """LangChain-compatible Embeddings wrapper gọi Core Embedding API (GPU microservice)"""

    def __init__(self, api_url: str = None, model_name: str = None):
        self.api_url = api_url or os.getenv("CORE_EMBED_URL", "http://core_embedding:8004")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_GPU", "intfloat/multilingual-e5-large")
        print(f"[CoreEmbeddings] Kết nối Core Embedding API tại: {self.api_url}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed danh sách texts qua Core API"""
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.api_url}/embed",
                    json={"inputs": texts}
                )
                response.raise_for_status()
                data = response.json()
                return data["embeddings"]
        except Exception as e:
            print(f"[CoreEmbeddings] ⚠️ Lỗi gọi Core Embedding API: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed 1 query qua Core API"""
        results = self.embed_documents([text])
        return results[0]


class Embedder:
    """Singleton Embedder - gọi Core Embedding API (GPU microservice)"""

    _instance: Optional["Embedder"] = None
    _embedding_model: Optional[Embeddings] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Chỉ chạy 1 lần khi tạo instance đầu tiên"""
        self.model_name = os.getenv("EMBEDDING_MODEL_GPU", "intfloat/multilingual-e5-large")
        core_url = os.getenv("CORE_EMBED_URL", "http://core_embedding:8004")
        print(f"[Embedder] Khởi tạo kết nối tới Core Embedding API tại: {core_url}")
        self._embedding_model = CoreEmbeddings(api_url=core_url, model_name=self.model_name)

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        return self._embedding_model.embed_documents([doc.page_content for doc in documents])

    def get_embedding_model(self) -> Embeddings:
        """Trả về model để truyền vào VectorStore"""
        return self._embedding_model

    def embed_query(self, text: str) -> List[float]:
        """Required by LangChain Embeddings interface"""
        return self._embedding_model.embed_query(text)