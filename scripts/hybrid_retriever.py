import os
from typing import List, Optional
from langchain_core.documents import Document

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from app.services.retrieval import RetrievalService


class HybridRetriever:
    """
    Hybrid Retriever (BM25 + Dense) - ĐÃ FIX để luôn trả về contexts
    """

    def __init__(self, top_k: int = 8):
        self.top_k = top_k
        self.hybrid_retriever = None
        self._initialized = False

    def _initialize(self):
        """Khởi tạo Hybrid Retriever một lần"""
        if self._initialized:
            return

        try:
            print("🔧 Đang khởi tạo Hybrid Retriever (BM25 + Dense)...")

            # Lấy RetrievalService để truy cập vectorstore
            retrieval_service = RetrievalService()

            # Lấy toàn bộ documents + metadata từ Chroma
            vectorstore = retrieval_service.vector_store._get_or_create_vectorstore(retrieval_service.embedder)

            if vectorstore is None:
                raise RuntimeError("VectorStore chưa được load. Vui lòng chạy ingestion trước!")

            # Lấy đầy đủ data từ Chroma
            data = vectorstore.get()
            all_page_contents = data.get("documents", [])
            all_metadatas = data.get("metadatas", [])

            if not all_page_contents:
                raise RuntimeError("VectorStore không có dữ liệu nào! Vui lòng chạy ingestion trước.")

            print(f"📦 Tổng documents trong VectorStore: {len(all_page_contents)}")

            # Tạo list Document chuẩn (có cả metadata)
            langchain_docs = []
            for content, meta in zip(all_page_contents, all_metadatas):
                langchain_docs.append(Document(
                    page_content=content,
                    metadata=meta or {}
                ))

            # 1. BM25 Retriever
            self.bm25_retriever = BM25Retriever.from_documents(
                langchain_docs,
                k=self.top_k * 2
            )
            print("bm25")

            # 2. Dense Retriever
            self.dense_retriever = retrieval_service.vector_store.get_retriever(
                search_kwargs={"k": self.top_k * 2}
            )

            print("bm25")

            # 3. Hybrid Retriever
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.dense_retriever],
                weights=[0.7, 0.3]          # Có thể chỉnh sau: [0.6, 0.4] hoặc [0.7, 0.3]
            )

            self._initialized = True
            print("✅ Hybrid Retriever đã khởi tạo thành công!")
        except Exception as e:
            self.bm25_retriever = None
            self.dense_retriever = None
            self.hybrid_retriever = None
            self._initialized = False
            raise

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Trả về documents từ Hybrid Retriever"""
        if not self._initialized:
            self._initialize()

        k = top_k or self.top_k

        print(f"🔎 Hybrid Retrieval cho query: '{query}' (top_k={k})")

        docs = self.hybrid_retriever.invoke(query)

        print(f"   → Hybrid trả về {len(docs)} documents")

        # Lấy top_k tốt nhất
        return docs[:k]