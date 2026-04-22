import os
from typing import List, Optional
from dotenv import load_dotenv 

from concurrent.futures import ThreadPoolExecutor

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import asyncio

load_dotenv()

_EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", min(4, (os.cpu_count() or 2))))
_embed_executor = ThreadPoolExecutor(
    max_workers=_EMBED_WORKERS,
    thread_name_prefix="embed-worker"
)

class Embedder(Embeddings):
    """"""
    
    _instance: Optional["Embedder"] = None
    _embedding_model: Optional[Embeddings] = None

    _sem: Optional[Embeddings] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Chỉ chạy 1 lần khi tạo instance đầu tiên"""
        self.device = os.getenv("EMBED_DEVICE", "cpu")
        self.model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        
        max_concurrent = int(os.getenv("EMBED_MAX_CONCURRENT",_EMBED_WORKERS))

        print(
            f"[Embedder] model={self.model_name} | "
            f"device={self.device.upper()} | "
            f"workers={_EMBED_WORKERS} | "
            f"max_concurrent={max_concurrent}"
        )
        
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )

        self._max_concurrent = max_concurrent

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max_concurrent)
        return self._sem

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        return self._embedding_model.embed_documents([doc.page_content for doc in documents])

    def get_embedding_model(self) -> Embeddings:
        """Trả về model để truyền vào VectorStore"""
        return self._embedding_model
    
    def embed_query(self, text: str) -> List[float]:
        """Required by LangChain Embeddings interface"""
        return self._embedding_model.embed_query(text)
    
    async def aembed_query(self, text: str) -> List[float]:
        loop  = asyncio.get_running_loop()
        async with self._get_semaphore():
            return await loop.run_in_executor(
                _embed_executor,
                self._embedding_model.embed_query,
                text,
            )
        
    async def aembed_documents(self, document: List[Document]) -> List[List[float]]:
        texts = [doc.page_content for doc in document]
        loop  = asyncio.get_running_loop()
        async with self._get_semaphore():
            return await loop.run_in_executor(
                _embed_executor,
                self._embedding_model.embed_documents,
                texts,
            )