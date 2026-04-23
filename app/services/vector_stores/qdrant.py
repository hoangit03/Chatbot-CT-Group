"""
Qdrant Vector Store
====================
Thay thế ChromaDB bằng Qdrant — nhanh hơn đáng kể với HNSW index,
hỗ trợ filtering phức tạp, và có REST/gRPC API native async.

ENV VARS:
    QDRANT_HOST             → localhost
    QDRANT_PORT             → 6333
    QDRANT_GRPC_PORT        → 6334
    QDRANT_API_KEY          → (optional, để trống nếu local)
    QDRANT_USE_GRPC         → false  (true để dùng gRPC — nhanh hơn với payload lớn)
    QDRANT_COLLECTION       → internal_knowledge
    QDRANT_WORKERS          → 4
    QDRANT_MAX_CONCURRENT   → 4

GHI CHÚ VỀ PERFORMANCE SO VỚI CHROMA:
    - Qdrant dùng HNSW (Hierarchical Navigable Small World) index → search O(log n)
    - Hỗ trợ payload filter trước/sau khi search → giảm số doc cần score
    - gRPC mode giảm latency thêm ~30% với document payloads lớn
    - Qdrant server là single process Rust — không cần thread pool như Chroma HTTP
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import asyncio

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.services.vector_stores.base import BaseVectorStore

load_dotenv()

_QDRANT_WORKERS = int(os.getenv("QDRANT_WORKERS", 4))
_qdrant_executor = ThreadPoolExecutor(
    max_workers=_QDRANT_WORKERS,
    thread_name_prefix="qdrant-worker",
)


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant implementation của BaseVectorStore.

    Thread-safety: _get_or_create_client() giữ một QdrantClient duy nhất per instance.
    Qdrant client là thread-safe — nhiều thread có thể dùng chung.
    """

    def __init__(
        self,
        collection_name: str = None,
        vector_size: int = None,
    ):
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.grpc_port = int(os.getenv("QDRANT_GRPC_PORT", 6334))
        self.api_key = os.getenv("QDRANT_API_KEY") or None
        self.use_grpc = os.getenv("QDRANT_USE_GRPC", "false").lower() == "true"
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "internal_knowledge")
        self._vector_size = vector_size  # None = auto-detect từ embedding model
        self._max_concurrent = int(os.getenv("QDRANT_MAX_CONCURRENT", _QDRANT_WORKERS))

        self._client = None          # qdrant_client.QdrantClient
        self._vectorstore = None     # langchain_qdrant.QdrantVectorStore
        self._embedding: Optional[Embeddings] = None
        self._sem: Optional[asyncio.Semaphore] = None

        logger = __import__("logging").getLogger(__name__)
        logger.info(
            f"[Qdrant] host={self.host}:{self.port} | "
            f"collection={self.collection_name} | grpc={self.use_grpc}"
        )

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max_concurrent)
        return self._sem

    # ─────────────────────────── PRIVATE HELPERS ───────────────────────────

    def _get_or_create_client(self):
        """Tạo QdrantClient singleton. Thread-safe sau lần đầu init."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError(
                    "qdrant-client chưa cài. Chạy: pip install qdrant-client langchain-qdrant"
                )
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                api_key=self.api_key,
                prefer_grpc=self.use_grpc,
            )
        return self._client

    def _get_or_create_vectorstore(self, embedding: Embeddings):
        """
        Load collection đã tồn tại hoặc tạo mới nếu chưa có.
        Langchain-Qdrant tự detect vector size từ embedding model.
        """
        if self._vectorstore is None or self._embedding != embedding:
            try:
                from langchain_qdrant import QdrantVectorStore as LCQdrant
            except ImportError:
                raise ImportError(
                    "langchain-qdrant chưa cài. Chạy: pip install langchain-qdrant"
                )
            client = self._get_or_create_client()
            self._vectorstore = LCQdrant(
                client=client,
                collection_name=self.collection_name,
                embedding=embedding,
            )
            self._embedding = embedding
        return self._vectorstore

    def _generate_stable_id(self, doc: Document) -> str:
        """ID dựa trên source + content hash → idempotent ingest."""
        source = doc.metadata.get("source", "unknown")
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:16]
        return f"{source}__{content_hash}"

    def _ensure_collection_exists(self, embedding: Embeddings) -> None:
        """
        Tạo collection nếu chưa có.
        Qdrant cần biết vector_size trước khi insert — lấy từ embedding model.
        """
        from qdrant_client.models import Distance, VectorParams
        client = self._get_or_create_client()

        existing = [c.name for c in client.get_collections().collections]
        if self.collection_name not in existing:
            # Auto-detect vector size nếu không set
            vector_size = self._vector_size
            if vector_size is None:
                sample_vec = embedding.embed_query("test")
                vector_size = len(sample_vec)

            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"[Qdrant] Tạo collection '{self.collection_name}' | dim={vector_size}")

    def _delete_collection_sync(self) -> None:
        client = self._get_or_create_client()
        try:
            client.delete_collection(self.collection_name)
            print(f"[Qdrant] Đã xóa collection '{self.collection_name}'")
        except Exception:
            pass  # Không tồn tại — bỏ qua
        self._vectorstore = None
        self._embedding = None

    # ─────────────────────────── SYNC METHODS ───────────────────────────

    def _add_documents_sync(
        self,
        documents: List[Document],
        embedding: Embeddings,
        replace: bool = True,
    ) -> None:
        if not documents:
            print("[Qdrant] Không có document nào để thêm")
            return

        mode = "REPLACE" if replace else "INCREMENTAL"
        print(f"[Qdrant] {mode}: {len(documents)} documents...")

        if replace:
            self._delete_collection_sync()

        self._ensure_collection_exists(embedding)

        from langchain_qdrant import QdrantVectorStore as LCQdrant
        client = self._get_or_create_client()

        ids = [self._generate_stable_id(doc) for doc in documents]

        if replace:
            # from_documents tạo mới hoàn toàn
            self._vectorstore = LCQdrant.from_documents(
                documents=documents,
                embedding=embedding,
                url=f"http://{self.host}:{self.port}",
                api_key=self.api_key,
                collection_name=self.collection_name,
                ids=ids,
                force_recreate=True,
            )
            self._embedding = embedding
        else:
            vs = self._get_or_create_vectorstore(embedding)
            vs.add_documents(documents=documents, ids=ids)

        print(f"[Qdrant] Đã lưu {len(documents)} docs vào '{self.collection_name}'")

    def _similarity_search_sync(
        self,
        query_embedding: List[float],
        k: int,
        metadata_filter,
    ) -> List[Document]:
        if self._vectorstore is None:
            from app.services.embedder import Embedder
            self._get_or_create_vectorstore(Embedder().get_embedding_model())

        # Chuyển dict filter sang Qdrant Filter nếu cần
        qdrant_filter = self._build_qdrant_filter(metadata_filter)

        results = self._vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=qdrant_filter,
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
        print(f"[Qdrant] Retriever sẵn sàng (k={safe_kwargs.get('k')})")
        return retriever

    @staticmethod
    def _build_qdrant_filter(metadata_filter: Optional[dict]):
        """
        Chuyển dict filter đơn giản {"key": "value"} sang Qdrant Filter object.
        Trả về None nếu không có filter.
        """
        if not metadata_filter:
            return None
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                for k, v in metadata_filter.items()
            ]
            return Filter(must=conditions)
        except ImportError:
            return None

    # ─────────────────────────── PUBLIC SYNC ───────────────────────────

    def delete_collection(self) -> None:
        self._delete_collection_sync()

    def add_documents(self, documents, embedding, replace) -> None:
        self._add_documents_sync(documents, embedding, replace)

    def get_retriever(self, search_kwargs=None):
        return self._get_retriever_sync(search_kwargs)

    # ─────────────────────────── ASYNC METHODS ───────────────────────────

    async def aadd_documents(
        self,
        documents: List[Document],
        embedding: Embeddings,
        replace: bool = True,
    ) -> None:
        loop = asyncio.get_running_loop()
        async with self._get_sem():
            await loop.run_in_executor(
                _qdrant_executor,
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
        Async search với pre-computed embedding.
        Qdrant gRPC mode giảm latency thêm ~30% — bật bằng QDRANT_USE_GRPC=true.
        """
        loop = asyncio.get_running_loop()
        async with self._get_sem():
            return await loop.run_in_executor(
                _qdrant_executor,
                self._similarity_search_sync,
                query_embedding,
                k,
                metadata_filter,
            )

    async def aget_retriever(self, search_kwargs: Optional[dict] = None):
        loop = asyncio.get_running_loop()
        async with self._get_sem():
            return await loop.run_in_executor(
                _qdrant_executor,
                self._get_retriever_sync,
                search_kwargs,
            )