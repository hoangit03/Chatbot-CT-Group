from typing import List, Union
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document

from app.utils.multi_document_loader import MultiDocumentLoader
from app.utils.preprocessor import TextPreprocessor
from app.services.document_splitter import RecursiveSplitter
from app.services.embedder import Embedder
from app.services.vector_stores.factory import VectorStoreFactory

load_dotenv()

class IngestionService:
    """Service chính chịu trách nhiệm ingestion - hỗ trợ Full & Incremental"""

    def __init__(self):
        self.loader = MultiDocumentLoader(raw_dir="data/raw")
        self.splitter = RecursiveSplitter()
        self.embedder = Embedder()
        self.preprocessor = TextPreprocessor()
        self.vector_store = VectorStoreFactory.get_vector_store()

    # ====================== FULL INGEST (chạy khi rebuild toàn bộ) ======================
    def run_full(self, replace: bool = True) -> None:
        """Ingest TOÀN BỘ data/raw - chỉ dùng khi muốn rebuild knowledge base"""
        print("BẮT ĐẦU FULL INGESTION (Toàn bộ data/raw)\n")

        print("Đang load tất cả documents từ data/raw...")
        docs = self.loader.load_all()
        if not docs:
            print("Không tìm thấy file nào!")
            return

        self._process_and_store(docs, replace=replace)

    # ====================== INCREMENTAL INGEST (dùng cho API upload) ======================
    def ingest_documents(self, file_paths: List[Union[str, Path]]) -> None:
        """
        Chỉ ingest một hoặc vài file cụ thể (rất nhanh)
        Dùng khi admin upload file mới qua API
        """
        print(f"BẮT ĐẦU INCREMENTAL INGESTION ({len(file_paths)} file)\n")

        if not file_paths:
            print("Không có file nào để ingest")
            return

        # Load chỉ những file được chỉ định (đã được tối ưu trong MultiDocumentLoader)
        docs = self.loader.load_files(file_paths)

        if not docs:
            print("Không load được document nào từ các file cung cấp")
            return

        self._process_and_store(docs, replace=False)

    # ====================== PRIVATE METHOD ======================
    def _process_and_store(self, docs: List[Document], replace: bool = True):
        """Xử lý chung cho cả full và incremental"""
        print(f"Đang chunking {len(docs)} documents...")
        docs = self.preprocessor.process_documents(docs)
        splits = self.splitter.split(docs)
        print(f"   → Tạo {len(splits)} chunks")

        print("Đang embedding và lưu vào Vector Store...")
        embedding_model = self.embedder.get_embedding_model()

        self.vector_store.add_documents(
            documents=splits,
            embedding=embedding_model,
            replace=replace
        )

        print(f"INGESTION HOÀN TẤT!")
        print(f"   Mode: {'FULL REPLACE' if replace else 'INCREMENTAL'}")
        print(f"   Số chunks: {len(splits)}")
        print(f"   Vector Store type: {os.getenv('VECTOR_STORE_TYPE', 'chroma')}")