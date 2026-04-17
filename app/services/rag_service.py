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

        retrieval_result = self.retrieval.retrieve(query=query)

        answer = self.generation.generate(retrieval_result, chat_history)

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