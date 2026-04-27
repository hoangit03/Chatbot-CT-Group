import os
import hashlib
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.services.vector_stores.base import BaseVectorStore

load_dotenv()

class ChromaVectorStore(BaseVectorStore):
    """
    Implementation hoàn chỉnh cho ChromaDB
    """
    def __init__(self, persist_dir: str = None, collection_name: str = "general"):
        self.persist_dir = persist_dir or  os.getenv("VECTOR_DB_DIR", "./vectorstore/chroma_db")
        self.collection_name = collection_name
        
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        self._vectorstore: Optional[Chroma] = None
        self._embedding: Optional[Embeddings] = None

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

    def delete_collection(self) -> None:
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
    def add_documents(
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
            self.delete_collection()
        else:
            print(f"INCREMENTAL MODE: Thêm {len(documents)} documents mới...")

        vectorstore = self._get_or_create_vectorstore(embedding)

        # Tạo ID 
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
                ids=ids
            )
        else:
            # Thêm incremental
            vectorstore.add_documents(documents=documents, ids=ids)

        # Khác với Local DB, HTTP Client tự động persist
        print(f"Đã lưu thành công {len(documents)} documents vào collection '{self.collection_name}'")

    def get_retriever(self, search_kwargs: Optional[dict] = None):
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