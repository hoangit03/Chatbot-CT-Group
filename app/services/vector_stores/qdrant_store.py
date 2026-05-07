import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore as LangchainQdrant
from qdrant_client import QdrantClient

from app.services.vector_stores.base import BaseVectorStore
from app.services.embedder import Embedder

class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant Vector Store Implementation for Retrieval.
    Data insertion is handled separately by the Shared ETL Service.
    """

    def __init__(self):
        host = os.getenv("QDRANT_HOST", "core_qdrant")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        # Determine the collection to query based on env var (or dynamic later)
        # Defaulting to env var, but later RetrievalService will pass dynamic filter
        self.collection_name = os.getenv("QDRANT_COLLECTION", "qtqd")
        
        self.client = QdrantClient(host=host, port=port)
        self.embedder = Embedder()
        
        # Use Langchain's Qdrant wrapper
        self.vectorstore = LangchainQdrant(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedder
        )

    def add_documents(self, documents: List[Document], embedding: Embeddings, replace: bool) -> None:
        """Not implemented: ETL is handled by Shared ETL Service"""
        pass

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Returns the vector store retriever with specific search kwargs"""
        kwargs = search_kwargs or {}
        # Ensure k is set
        if "k" not in kwargs:
            kwargs["k"] = 5
            
        return self.vectorstore.as_retriever(search_kwargs=kwargs)

    def delete_collection(self) -> None:
        """Not implemented: handled by Shared ETL Service"""
        pass
