import os
from typing import List, Optional
from dotenv import load_dotenv 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

load_dotenv()

class Embedder:
    """"""
    
    _instance: Optional["Embedder"] = None
    _embedding_model: Optional[Embeddings] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Chỉ chạy 1 lần khi tạo instance đầu tiên"""
        self.device = os.getenv("DEVICE", "cpu")
        self.model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        
        print(f"[Embedder] Khởi tạo model: {self.model_name} | Device: {self.device.upper()}")
        
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        return self.embedding_model.embed_documents([doc.page_content for doc in documents])

    def get_embedding_model(self) -> Embeddings:
        """Trả về model để truyền vào VectorStore"""
        return self._embedding_model