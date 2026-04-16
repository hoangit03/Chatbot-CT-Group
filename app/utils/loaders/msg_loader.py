from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.utils.base.document_loader import DocumentLoader


class MSGLoader(DocumentLoader):
    """Loader cho file .msg (Outlook email)"""
    def load(self, file_path: Path) -> List[Document]:
        def _load():
            try:
                from extract_msg import Message
                msg = Message(str(file_path))

                text = f"Subject: {msg.subject or 'No subject'}\n"
                text += f"From: {msg.sender or 'Unknown'}\n"
                text += f"To: {msg.to or 'Unknown'}\n"
                text += f"Date: {msg.date or 'Unknown'}\n"
                text += f"CC: {msg.cc or ''}\n\n"
                text += msg.body or msg.htmlBody or "[No content]"

                metadata = self._create_metadata(file_path, "msg")
                return [Document(page_content=text, metadata=metadata)]

            except Exception as e:
                return [Document(
                    page_content=f"[Lỗi đọc file .msg: {str(e)}]\nFile: {file_path.name}",
                    metadata=self._create_metadata(file_path, "msg")
                )]
        return self._safe_load(file_path, _load)