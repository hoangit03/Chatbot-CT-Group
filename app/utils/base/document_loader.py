from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.core.exception.loader_exception import DocumentLoadError, InvalidFileError

class DocumentLoader(ABC):
    """Abstract base class theo SOLID - Open/Closed Principle"""

    @abstractmethod
    def load(self, file_path: Path) -> List[Document]:
        """Mỗi loader con chỉ chịu trách nhiệm load 1 loại file"""
        pass

    def _create_metadata(self, file_path: Path, file_type: str) -> dict:
        """Helper chung để tạo metadata (dùng cho citation sau này)"""
        return {
            "source": str(file_path.relative_to(Path("data/raw").resolve())),
            "file_type": file_type,
            "file_name": file_path.name,
        }
    
    def _safe_load(self, file_path: Path, load_func) -> List[Document]:
        """Helper chung để catch lỗi và raise custom exception"""
        try:
            return load_func()
        except Exception as e:
            if "not found" in str(e).lower() or "permission" in str(e).lower():
                raise InvalidFileError(file_path, e) from e
            raise DocumentLoadError(file_path, e) from e