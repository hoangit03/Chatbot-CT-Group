from pathlib import Path
from typing import Dict, Type

from app.utils.base.document_loader import DocumentLoader
from app.utils.loaders.pdf_loader import PDFLoader
from app.utils.loaders.docx_loader import DOCXLoader
from app.utils.loaders.excel_loader import ExcelLoader
from app.utils.loaders.pptx_loader import PPTXLoader
from app.utils.loaders.doc_loader import DOCLoader
from app.utils.loaders.msg_loader import MSGLoader
from app.core.exception.loader_exception import UnsupportedFileExtensionError

class DocumentLoaderFactory:
    """Factory + Registry theo Dependency Inversion Principle"""

    _loaders: Dict[str, Type[DocumentLoader]] = {}

    @classmethod
    def register_loader(cls, extension: str, loader_class: Type[DocumentLoader]):
        """Đăng ký loader mới (dùng để mở rộng sau này)"""
        cls._loaders[extension.lower()] = loader_class

    @classmethod
    def get_loader(cls, file_path: Path) -> DocumentLoader:
        ext = file_path.suffix.lower()
        if ext not in cls._loaders:
            raise UnsupportedFileExtensionError(file_path)
        return cls._loaders[ext]()

    @classmethod
    def register_default_loaders(cls):
        """Đăng ký các loader mặc định"""
        cls.register_loader(".pdf", PDFLoader)
        cls.register_loader(".docx", DOCXLoader)
        # cls.register_loader(".xlsx", ExcelLoader)
        # cls.register_loader(".xlsm", ExcelLoader)
        # cls.register_loader(".pptx", PPTXLoader)
        # cls.register_loader(".doc", DOCLoader)
        # cls.register_loader(".msg", MSGLoader)


# Khởi tạo các loader
DocumentLoaderFactory.register_default_loaders()