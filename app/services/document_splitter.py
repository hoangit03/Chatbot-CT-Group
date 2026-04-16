from abc import ABC, abstractmethod
from typing import List
from dotenv import load_dotenv 
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class DocumentSplitter(ABC):
    """Abstract class cho việc chunking"""
    
    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        pass


class RecursiveSplitter(DocumentSplitter):
    """Concrete implementation sử dụng RecursiveCharacterTextSplitter"""
    
    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)