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

_CHROMA_WORKERS = int(os.getenv("CHROMA_WORKERS", 4))
_chroma_executor = ThreadPoolExecutor(
    max_workers=_CHROMA_WORKERS,
    thread_name_prefix="chroma-worker"
)

load_dotenv()

class ChromaVectorStore(BaseVectorStore):
    """
    Implementation hoàn chỉnh cho ChromaDB
    """

    def __init__(self, persist_dir: str = None, collection_name: str = "internal_knowledge"):
        self.persist_dir = persist_dir or  os.getenv("VECTOR_DB_DIR", "./vectorstore/chroma_db")
        self.collection_name = collection_name
        
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        self._vectorstore: Optional[Chroma] = None
        self._embedding: Optional[Embeddings] = None

        max_concurrent = int(os.getenv("CHROMA_MAX_CONCURRENT", _CHROMA_WORKERS))
        self._sem = asyncio.Semaphore(max_concurrent)

    # ====================== HELPER METHODS ======================
    def _get_or_create_vectorstore(self, embedding: Embeddings) -> Chroma:
        """Load DB đã tồn tại hoặc tạo mới"""
        if self._vectorstore is None or self._embedding != embedding:
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
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
        if Path(self.persist_dir).exists():
            import shutil
            shutil.rmtree(self.persist_dir)
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        self._vectorstore = None
        self._embedding = None
        print(f"Đã xóa toàn bộ collection '{self.collection_name}'")

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

        if replace:
            print(f"REPLACE MODE: Xóa DB cũ và tạo mới với {len(documents)} documents...")
            self._delete_collection_sync()
        else:
            print(f"INCREMENTAL MODE: Thêm {len(documents)} documents mới...")

        # Tạo ID 
        ids = [self._generate_stable_id(doc) for doc in documents]

        if replace:
            # Tạo mới hoàn toàn
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embedding,
                persist_directory=self.persist_dir,
                collection_name=self.collection_name,
                ids=ids
            )
        else:
            # Thêm incremental
            vs = self._get_or_create_vectorstore(embedding)
            vs.add_documents(documents=documents, ids=ids)

        print(f"Đã lưu thành công {len(documents)} documents vào collection '{self.collection_name}'")

    def _similarity_search_sync(
        self,
        query_embedding: List[float],
        k: int,
        metadata_filter: Optional[dict],      
    )-> List[Document]:
        vs = self._vectorstore
        if vs is None:
            raise RuntimeError("VectorStore chưa được khởi tạo. Gọi get_retriever trước.")
 
        results = vs.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=metadata_filter,
        )
        return results

    def _get_retriever_sync(self, search_kwargs: Optional[dict] = None):
        """Trả về retriever đã sẵn sàng sử dụng cho Retrieval"""
        if not self._vectorstore:
            from app.services.embedder import Embedder
            embedding_model = Embedder().get_embedding_model()
            self._get_or_create_vectorstore(embedding_model)

        safe_kwargs = {}
        if search_kwargs:
            safe_kwargs.update({k: v for k, v in search_kwargs.items() if k != "score_threshold"})

        retriever = self._vectorstore.as_retriever(search_kwargs=safe_kwargs)
        print(f"Retriever sẵn sàng (k={safe_kwargs.get('k')})")
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
        async with self._sem:
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
        async with self._sem:
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
        async with self._sem:
            return await loop.run_in_executor(
                _chroma_executor,
                self._get_retriever_sync,
                search_kwargs,
            )