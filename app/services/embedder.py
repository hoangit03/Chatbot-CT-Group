import os
from typing import List, Optional
from dotenv import load_dotenv 

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

load_dotenv()

class Embedder:
    """Singleton Embedder - Hỗ trợ TEI Microservice hoặc chạy Local Model"""
    
    _instance: Optional["Embedder"] = None
    _embedding_model: Optional[Embeddings] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Chỉ chạy 1 lần khi tạo instance đầu tiên"""
        self.tei_url = os.getenv("TEI_EMBEDDER_URL")
        if self.tei_url:
            print(f"[Embedder] Khởi tạo kết nối tới Custom Embedder API tại: {self.tei_url}")
            self.model_name = os.getenv("EMBEDDING_MODEL_GPU", "intfloat/multilingual-e5-large")
            self._embedding_model = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_base=self.tei_url,
                openai_api_key="empty",
                check_embedding_ctx_length=False
            )
        else:
            self.device = os.getenv("EMBED_DEVICE", "cuda").lower()
            
            # Chọn model dựa trên device
            if self.device == "cuda":
                self.model_name = os.getenv("EMBEDDING_MODEL_GPU", "intfloat/multilingual-e5-large")
            else:
                self.model_name = os.getenv("EMBEDDING_MODEL_CPU", "paraphrase-multilingual-MiniLM-L12-v2")
            
            print(f"[Embedder] Khởi tạo model LOCAL: {self.model_name} | Device: {self.device.upper()}")
            
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        return self._embedding_model.embed_documents([doc.page_content for doc in documents])

    def get_embedding_model(self) -> Embeddings:
        """Trả về model để truyền vào VectorStore"""
        return self._embedding_model
    
    def embed_query(self, text: str) -> List[float]:
        """Required by LangChain Embeddings interface"""
        return self._embedding_model.embed_query(text)