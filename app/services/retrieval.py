import os
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
        docs = retriever.invoke(query)
 
        if score_threshold > 0.0:
            docs = [
                d for d in docs
                if d.metadata.get("similarity_score", 1.0) >= score_threshold
            ]
 
        if self.reranker and docs:
            scored: List[Tuple[Document, float]] = self.reranker.rerank(query, docs)
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