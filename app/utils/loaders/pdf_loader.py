from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.utils.base.document_loader import DocumentLoader


class PDFLoader(DocumentLoader):
    def load(self, file_path: Path) -> List[Document]:
        def _load():
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            metadata = self._create_metadata(file_path, "pdf")
            for doc in docs:
                doc.metadata.update(metadata)
            return docs
        return self._safe_load(file_path, _load)