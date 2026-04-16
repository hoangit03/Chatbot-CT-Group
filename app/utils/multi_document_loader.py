from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from langchain_core.documents import Document

from app.utils.document_loader_factory import DocumentLoaderFactory
from app.core.exception.loader_exception import LoaderException


class MultiDocumentLoader:
    """Loader chính – chỉ orchestrate"""

    SUPPORTED_EXT = {".pdf", ".docx", ".doc", ".xlsx", ".xlsm", ".pptx", ".msg"}

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir).resolve()
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Thư mục không tồn tại: {self.raw_dir}")

    # ====================== LOAD TOÀN BỘ ======================
    def load_all(self) -> List[Document]:
        """Load recursive tất cả file trong data/raw (dùng cho full ingestion)"""
        all_docs = []
        files = []

        for ext in self.SUPPORTED_EXT:
            files.extend(self.raw_dir.rglob(f"*{ext}"))

        print(f"Tìm thấy {len(files)} file hỗ trợ trong data/raw (và các subfolder)")

        for file_path in tqdm(files, desc="Đang load documents"):
            try:
                loader = DocumentLoaderFactory.get_loader(file_path)
                docs = loader.load(file_path)
                all_docs.extend(docs)
            except LoaderException as e:
                print(f"{type(e).__name__}: {e}")
                continue
            except Exception as e:
                print(f"Lỗi load file {file_path.name}: {str(e)}")
                continue

        print(f"Tổng cộng load được {len(all_docs)} documents")
        return all_docs

    # ====================== LOAD FILE CỤ THỂ (MỚI - DÙNG CHO API) ======================
    def load_files(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        Load chỉ những file được chỉ định (incremental ingest)
        """
        all_docs = []
        print(f"Đang load {len(file_paths)} file cụ thể...")

        for path in file_paths:
            file_path = Path(path).resolve()
            if not file_path.exists():
                print(f"File không tồn tại: {file_path}")
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXT:
                print(f"Định dạng không hỗ trợ: {file_path.suffix} ({file_path.name})")
                continue

            try:
                loader = DocumentLoaderFactory.get_loader(file_path)
                docs = loader.load(file_path)
                all_docs.extend(docs)
                print(f"    Loaded: {file_path.name} ({len(docs)} documents)")
            except LoaderException as e:
                print(f" {type(e).__name__}: {e} | File: {file_path.name}")
                continue
            except Exception as e:
                print(f" Lỗi load file {file_path.name}: {str(e)}")
                continue

        print(f" Tổng cộng load được {len(all_docs)} documents từ file cụ thể")
        return all_docs

    def load_single_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Tiện lợi cho trường hợp upload 1 file duy nhất"""
        return self.load_files([file_path])