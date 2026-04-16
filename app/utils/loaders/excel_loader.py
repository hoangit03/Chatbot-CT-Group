from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document

from app.utils.base.document_loader import DocumentLoader


class ExcelLoader(DocumentLoader):
    def load(self, file_path: Path) -> List[Document]:
        def _load():
            text = f"File: {file_path.name}\n\n"
            try:
                xl = pd.ExcelFile(str(file_path))
                for sheet_name in xl.sheet_names:
                    df = xl.parse(sheet_name)
                    text += f"--- Sheet: {sheet_name} ---\n"
                    text += df.to_string(index=False) + "\n\n"
            except Exception as e:
                text += f"[Lỗi đọc Excel: {str(e)}]\n"

            metadata = self._create_metadata(file_path, "excel")
            return [Document(page_content=text, metadata=metadata)]
        return self._safe_load(file_path, _load)