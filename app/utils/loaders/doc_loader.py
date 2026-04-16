from pathlib import Path
from typing import List

from langchain_core.documents import Document
import win32com.client

from app.utils.base.document_loader import DocumentLoader
from app.core.exception.loader_exception import DocumentLoadError 


class DOCLoader(DocumentLoader):
    """Loader cho file .doc (Word legacy) - Windows Native (win32com)"""

    def load(self, file_path: Path) -> List[Document]:
        def _load():
            word_app = None
            doc = None
            try:
                # Mở Microsoft Word (ẩn)
                word_app = win32com.client.Dispatch("Word.Application")
                word_app.Visible = False
                word_app.DisplayAlerts = False

                doc = word_app.Documents.Open(str(file_path.absolute()))

                text = doc.Content.Text.strip()

                metadata = self._create_metadata(file_path, "doc")

                return [Document(
                    page_content=text,
                    metadata=metadata
                )]

            except Exception as e:
                raise DocumentLoadError(file_path, e) from e

            finally:
                if doc is not None:
                    doc.Close(SaveChanges=False)
                if word_app is not None:
                    word_app.Quit()

        return self._safe_load(file_path, _load)