import re
from typing import List

from langchain_core.documents import Document


class TextPreprocessor:
    """Xử lý dữ liệu trước khi chunking - giảm noise, tăng chất lượng RAG"""

    def clean_text(self, text: str) -> str:
        # Loại bỏ ký tự thừa, khoảng trắng nhiều
        text = re.sub(r'\s+', ' ', text.strip())
    
        return text.strip()

    def process_documents(self, docs: List[Document]) -> List[Document]:
        """Xử lý toàn bộ documents"""
        print(f"Đang preprocess {len(docs)} documents...")
        processed = []
        
        for doc in docs:
            cleaned = self.clean_text(doc.page_content)
            # if len(cleaned) < 30:  # Bỏ document quá ngắn
            #     continue
            
            new_doc = Document(
                page_content=cleaned,
                metadata=doc.metadata.copy()
            )
            processed.append(new_doc)

        print(f"Preprocess xong: {len(processed)} documents (đã loại {len(docs) - len(processed)} document ngắn)")
        return processed