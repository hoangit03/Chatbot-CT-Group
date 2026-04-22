import time
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage

from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService


class RAGService:
    """RAG Service hoàn chỉnh với multi-turn conversation"""

    def __init__(self):
        self.retrieval = RetrievalService()
        self.generation = GenerationService()

    def answer(self, query: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
        chat_history = chat_history or []

        t_total = time.time()

        # ── Bước 1: Retrieval (Embedding + VectorDB + Rerank) ──
        t0 = time.time()
        retrieval_result = self.retrieval.retrieve(query=query)
        t_retrieval = time.time() - t0

        # ── Bước 2: Generation (LLM Inference) ──
        t1 = time.time()
        answer = self.generation.generate(retrieval_result, chat_history)
        t_generation = time.time() - t1

        t_pipeline = time.time() - t_total

        # ── Debug Timing Report ──
        print(f"\n{'='*60}")
        print(f"  ⏱️  PIPELINE TIMING REPORT")
        print(f"{'='*60}")
        print(f"  📥 Retrieval (Embed+Search+Rerank) : {t_retrieval:.2f}s")
        print(f"  🤖 LLM Generation                  : {t_generation:.2f}s")
        print(f"  ──────────────────────────────────────────────")
        print(f"  🏁 TỔNG PIPELINE                    : {t_pipeline:.2f}s")
        print(f"{'='*60}\n")

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "file_name": doc.metadata.get("source_file") or doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown",
                    "score": doc.metadata.get("rerank_score") or doc.metadata.get("similarity_score", 0)
                }
                for doc in retrieval_result.documents
            ],
            "retrieved_count": len(retrieval_result.documents)
        }