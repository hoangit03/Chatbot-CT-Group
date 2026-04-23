from abc import ABC, abstractmethod
from typing import List
from dotenv import load_dotenv 
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Custom separators: ưu tiên cắt tại ranh giới mục tiếng Việt
_VIETNAMESE_SEPARATORS = [
    "\n\n",           # Đoạn văn
    "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. ",
    "\n6. ", "\n7. ", "\n8. ", "\n9. ",
    "\n* ",            # Bullet points
    "\n+ ",            # Sub-bullets
    "\n- ",            # Dash bullets
    "\n",              # Dòng mới
    ". ",              # Câu
    " ",               # Từ
    "",                # Fallback
]


class DocumentSplitter(ABC):
    """Abstract class cho việc chunking"""
    
    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        pass


class RecursiveSplitter(DocumentSplitter):
    """Concrete implementation sử dụng RecursiveCharacterTextSplitter"""
    
    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1500))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 250))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=_VIETNAMESE_SEPARATORS,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)