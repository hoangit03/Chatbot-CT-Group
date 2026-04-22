from typing import List, Dict, Any
from langchain_core.messages import BaseMessage
import asyncio

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
        return self._build_response(query, answer, retrieval_result)
    
    async def aanswer(
        self, query: str, chat_history: List[BaseMessage] = None
    ) -> Dict[str, Any]:
        """
        """
        chat_history = chat_history or []
 
        retrieval_result = await self.retrieval.aretrieve(query=query)
        answer = await self.generation.agenerate(retrieval_result, chat_history)
 
        return self._build_response(query, answer, retrieval_result)

    @staticmethod
    def _build_response(query, answer, retrieval_result) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "file_name": doc.metadata.get("file_name"),
                    "score": doc.metadata.get("rerank_score")
                    or doc.metadata.get("similarity_score", 0),
                }
                for doc in retrieval_result.documents
            ],
            "retrieved_count": len(retrieval_result.documents),
        }