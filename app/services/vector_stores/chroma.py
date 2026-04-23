import os
import hashlib
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import asyncio

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.services.vector_stores.base import BaseVectorStore

load_dotenv()

_CHROMA_WORKERS = int(os.getenv("CHROMA_WORKERS", 4))
_chroma_executor = ThreadPoolExecutor(
    max_workers=_CHROMA_WORKERS,
    thread_name_prefix="chroma-worker"
)

class ChromaVectorStore(BaseVectorStore):
    """
    Implementation hoàn chỉnh cho ChromaDB
    """

    def __init__(self, persist_dir: str = None, collection_name: str = "internal_knowledge"):
        self.persist_dir = persist_dir or os.getenv(
            "VECTOR_DB_DIR", "./vectorstore/chroma_db"
        )
        self.collection_name = collection_name
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
 
        self._vectorstore: Optional[Chroma] = None
        self._embedding: Optional[Embeddings] = None
 
        self._max_concurrent = int(os.getenv("CHROMA_MAX_CONCURRENT", _CHROMA_WORKERS))
        self._sem: Optional[asyncio.Semaphore] = None

    def _get_sem(self) -> asyncio.Semaphore:
        """Lazy-init semaphore trong running event loop."""
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max_concurrent)
        return self._sem
    # ====================== HELPER METHODS ======================
    def _get_or_create_vectorstore(self, embedding: Embeddings) -> Chroma:
        """Load DB đã tồn tại hoặc tạo mới"""
        import chromadb
        if self._vectorstore is None or self._embedding != embedding:
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = int(os.getenv("CHROMA_PORT", 8002))
            http_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            self._vectorstore = Chroma(
                client=http_client,
                embedding_function=embedding,
                collection_name=self.collection_name
            )
            self._embedding = embedding
        return self._vectorstore

    def _generate_stable_id(self, doc: Document) -> str:
        """Tạo ID duy nhất dựa trên source + nội dung → tránh duplicate khi add incremental"""
        source = doc.metadata.get("source", "unknown")
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:16]
        return f"{source}::{content_hash}"

    def _delete_collection_sync(self) -> None:
        """Xóa toàn bộ collection (dùng khi replace=True)"""
        import chromadb
        try:
            http_client = chromadb.HttpClient(host="localhost", port=8002)
            http_client.delete_collection(name=self.collection_name)
        except Exception as e:
            pass # Lỗi nếu collection chưa tồn tại
        
        self._vectorstore = None
        self._embedding = None
        print(f"Đã xóa toàn bộ collection '{self.collection_name}' qua HTTP API")

    # ====================== MAIN METHODS ======================
    def _add_documents_sync(
        self, 
        documents: List[Document], 
        embedding: Embeddings, 
        replace: bool = True
    ) -> None:
        """
        Thêm documents vào Vector Store
        
        Parameters:
            replace=True  → Xóa DB cũ rồi tạo mới (dùng cho ingest toàn bộ)
            replace=False → Chỉ thêm documents mới (incremental - nhanh)
        """
        if not documents:
            print("Không có document nào để thêm")
            return

        mode = "REPLACE" if replace else "INCREMENTAL"
        print(f"[Chroma] {mode}: {len(documents)} documents...")
 
        if replace:
            self._delete_collection_sync()
 
        ids = [self._generate_stable_id(doc) for doc in documents]
 
        if replace:
            import chromadb
            http_client = chromadb.HttpClient(host="localhost", port=8002)
            # Tạo mới hoàn toàn
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embedding,
                client=http_client,
                collection_name=self.collection_name,
                ids=ids,
            )
        else:
            vs = self._get_or_create_vectorstore(embedding)
            vs.add_documents(documents=documents, ids=ids)
 
        print(f"[Chroma] Đã lưu {len(documents)} docs vào '{self.collection_name}'")

    def _similarity_search_sync(
        self,
        query_embedding: List[float],
        k: int,
        metadata_filter: Optional[dict],      
    )-> List[Document]:
        if self._vectorstore is None:
            from app.services.embedder import Embedder
            self._get_or_create_vectorstore(Embedder().get_embedding_model())
        vs = self._vectorstore
        results = vs.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=metadata_filter,
        )
        return results


    def _get_retriever_sync(self, search_kwargs: Optional[dict] = None):
        if not self._vectorstore:
            from app.services.embedder import Embedder
            self._get_or_create_vectorstore(Embedder().get_embedding_model())
 
        safe_kwargs = {
            k: v for k, v in (search_kwargs or {}).items()
            if k != "score_threshold"
        }
        retriever = self._vectorstore.as_retriever(search_kwargs=safe_kwargs)
        print(f"[Chroma] Retriever sẵn sàng (k={safe_kwargs.get('k')})")
        return retriever
    
    def delete_collection(self) -> None:
        self._delete_collection_sync()
    
    def add_documents(self, documents, embedding, replace) -> None:
        self._add_documents_sync(
            documents, 
            embedding, 
            replace
        )
    
    def get_retriever(self, search_kwargs = None):
        return self._get_retriever_sync(search_kwargs)
    
    
    async def aadd_documents(
        self,
        documents: List[Document],
        embedding: Embeddings,
        replace: bool = True,
    ) -> None:
        """Bất đồng bộ — offload I/O-heavy ingest sang executor."""
        loop = asyncio.get_running_loop()
        async with self._get_sem():
            await loop.run_in_executor(
                _chroma_executor,
                self._add_documents_sync,
                documents,
                embedding,
                replace,
            )

    async def asimilarity_search(
        self,
        query_embedding: List[float],
        k: int = 6,
        metadata_filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Bất đồng bộ — nhận vector đã embed, tìm kiếm trong Chroma.
 
        Tách embedding ra ngoài (đã xử lý ở Embedder.aembed_query)
        để 2 tác vụ có thể dùng thread pool riêng biệt.
        """
        loop = asyncio.get_running_loop()
        async with self._get_sem():
            return await loop.run_in_executor(
                _chroma_executor,
                self._similarity_search_sync,
                query_embedding,
                k,
                metadata_filter,
            )
 
    async def aget_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Bất đồng bộ — đảm bảo vectorstore được load trước khi serve.
        Gọi một lần lúc startup, sau đó dùng asimilarity_search trực tiếp.
        """
        loop = asyncio.get_running_loop()
        async with self._get_sem():
            return await loop.run_in_executor(
                _chroma_executor,
                self._get_retriever_sync,
                search_kwargs,
            )