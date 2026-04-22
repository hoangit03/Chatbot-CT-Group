import os
import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_core.documents import Document

from app.services.embedder import Embedder
from app.services.vector_stores.factory import VectorStoreFactory
from app.services.reranker.base import BaseReranker
from app.services.reranker.cross_encoder import CrossEncoderReranker

load_dotenv()

@dataclass
class RetrievalResult:
    documents: List[Document]
    query: str
    top_k: int
    total_retrieved: int
    reranked: bool = False


class RetrievalService:
    """Retrieval Service với hỗ trợ Reranker (SOLID)"""

    def __init__(self, embedder: Optional[Embedder] = None, reranker: Optional[BaseReranker] = None):
        self.top_k = int(os.getenv("RETRIEVAL_TOP_K",6))
        self.reranker_top_k = int(os.getenv("RERANKER_TOP_K",20))
        self.embedder = embedder or Embedder()
        self.vector_store = VectorStoreFactory.get_vector_store()
        # Khởi tạo reranker nếu được bật trong .env
        self.reranker: Optional[BaseReranker] = None
        if os.getenv("RERANKER_ENABLED", "false").lower() == "true":
            self.reranker = reranker or CrossEncoderReranker(
                model_name=os.getenv("RERANKER_MODEL")
            )

    def retrieve(self,
                 query: str,
                 score_threshold: float = 0.0,
                 metadata_filter: Optional[Dict] = None) -> RetrievalResult:
        """
        1. Vector retrieval
        2. (Optional) Rerank bằng Cross-Encoder
        """
        print(f"[Retrieval] sync | query='{query}' top_k={self.top_k}")
        initial_k = self.reranker_top_k if self.reranker else self.top_k

        retriever = self.vector_store.get_retriever(
            search_kwargs={"k": initial_k, "filter": metadata_filter}
        )

        t_search = time.time()
        docs = retriever.invoke(query)  
        t_search_done = time.time() - t_search
        print(f"  ⏱️  [Vector Search] {len(docs)} docs trong {t_search_done:.2f}s")

        # ── DEBUG: Hiển thị dữ liệu trích xuất từ VectorDB ──
        if docs:
            print(f"\n  {'─'*70}")
            print(f"  📚 DỮ LIỆU TRÍCH XUẤT TỪ VECTORDB ({len(docs)} chunks):")
            print(f"  {'─'*70}")
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source_file") or doc.metadata.get("source") or doc.metadata.get("file_name") or "N/A"
                chunk_id = doc.metadata.get("chunk_id", "?")
                snippet = doc.page_content[:150].replace('\n', ' ')
                print(f"  [{i+1}] 📄 Nguồn: {source}")
                print(f"      🏷️  Chunk ID: {chunk_id}")
                print(f"      📝 Nội dung: \"{snippet}...\"")
                print()

        if score_threshold > 0.0:
            docs = [
                d for d in docs
                if d.metadata.get("similarity_score", 1.0) >= score_threshold
            ]
 
        if self.reranker and docs:
            t_rerank = time.time()
            print(f"  🔄 [Rerank] Đang chấm điểm {len(docs)} documents...")
            scored_docs: List[Tuple[Document, float]] = self.reranker.rerank(query, docs)
            t_rerank_done = time.time() - t_rerank
            print(f"  ⏱️  [Rerank] Hoàn tất trong {t_rerank_done:.2f}s")

            # ── DEBUG: Bảng xếp hạng Reranker ──
            print(f"\n  {'─'*70}")
            print(f"  🏆 BẢNG XẾP HẠNG SAU RERANK (Top {self.top_k} được chọn):")
            print(f"  {'─'*70}")
            print(f"  {'Hạng':<6} {'Score':<12} {'Nguồn':<40} {'Nội dung (50 ký tự)'}")
            print(f"  {'─'*70}")
            for rank, (doc, score) in enumerate(scored_docs):
                source = doc.metadata.get("source_file") or doc.metadata.get("source") or doc.metadata.get("file_name") or "N/A"
                snippet = doc.page_content[:50].replace('\n', ' ')
                marker = "✅" if rank < self.top_k else "  "
                print(f"  {marker} #{rank+1:<4} {score:<12.4f} {source:<40} \"{snippet}...\"")
            print(f"  {'─'*70}\n")

            # ── Lọc bỏ docs có score ÂM (không liên quan) ──
            positive_docs = [(doc, score) for doc, score in scored_docs if score > 0]
            if positive_docs:
                # Chỉ lấy những docs thật sự liên quan
                final_docs = [doc for doc, _ in positive_docs[:self.top_k]]
                for doc, score in positive_docs[:self.top_k]:
                    doc.metadata["rerank_score"] = float(score)
                print(f"  🧹 [Filter] Lọc: {len(positive_docs)} docs có score > 0 / {len(scored_docs)} tổng")
            else:
                # KHÔNG nhồi docs rác → trả rỗng để Generation dùng prompt SIMPLE
                final_docs = []
                print(f"  🚫 [Filter] Tất cả {len(scored_docs)} docs score âm → Trả rỗng (chống Hallucination)")
            reranked = True
        else:
            final_docs = docs[: self.top_k]
            reranked = False
 
        print(f"[Retrieval] trả về {len(final_docs)} docs {'(reranked)' if reranked else ''}")
        return RetrievalResult(
            documents=final_docs,
            query=query,
            top_k=self.top_k,
            total_retrieved=len(final_docs),
            reranked=reranked,
        )
    
    async def aretrieve(
        self,
        query: str,
        score_threshold: float = 0.0,
        metadata_filter: Optional[Dict] = None
    ) -> RetrievalResult:
        import asyncio
 
        print(f"[Retrieval] async | query='{query}' top_k={self.top_k}")
        initial_k = self.reranker_top_k if self.reranker else self.top_k
 
        # Step 1: embed query bất đồng bộ
        query_embedding = await self.embedder.aembed_query(query)
 
        # Step 2: vector search bất đồng bộ (nhận embedding, không embed lại)
        docs = await self.vector_store.asimilarity_search(
            query_embedding=query_embedding,
            k=initial_k,
            metadata_filter=metadata_filter,
        )
 
        if score_threshold > 0.0:
            docs = [
                d for d in docs
                if d.metadata.get("similarity_score", 1.0) >= score_threshold
            ]
 
        # Step 3: rerank (CPU-bound) — offload sang default thread pool
        if self.reranker and docs:
            scored: List[Tuple[Document, float]] = await self.reranker.arerank(query, docs)
            final_docs = [d for d, _ in scored[: self.top_k]]
            for d, score in scored[: self.top_k]:
                d.metadata["rerank_score"] = float(score)
            reranked = True
        else:
            final_docs = docs[: self.top_k]
            reranked = False
 
        print(f"[Retrieval] trả về {len(final_docs)} docs {'(reranked)' if reranked else ''}")
        return RetrievalResult(
            documents=final_docs,
            query=query,
            top_k=self.top_k,
            total_retrieved=len(final_docs),
            reranked=reranked,
        )
