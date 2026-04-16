from enum import Enum
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate


class PromptType(str, Enum):
    RAG = "rag"
    SIMPLE = "simple"

class PromptRegistry:
    """Registry cho phép dễ dàng thêm prompt mới mà không sửa code cũ"""

    _templates: Dict[PromptType, ChatPromptTemplate] = {}

    @classmethod
    def register(cls, prompt_type: PromptType, template: ChatPromptTemplate):
        cls._templates[prompt_type] = template

    @classmethod
    def get(cls, prompt_type: PromptType) -> ChatPromptTemplate:
        if prompt_type not in cls._templates:
            raise ValueError(f"Prompt type '{prompt_type}' chưa được đăng ký")
        return cls._templates[prompt_type]

    @classmethod
    def register_defaults(cls):
        # Prompt RAG chính
        rag_template = ChatPromptTemplate.from_messages([
            ("system", """Bạn là trợ lý thông minh của CT-Group.
Trả lời CHÍNH XÁC, NGẮN GỌN, rõ ràng bằng tiếng Việt.
Chỉ dùng thông tin từ Context được cung cấp.
Nếu không có thông tin → trả lời: "Tôi không tìm thấy thông tin này trong tài liệu nội bộ."
Luôn trích dẫn nguồn file khi trả lời."""),
            ("human", "Câu hỏi: {question}\n\nContext:\n{context}")
        ])

        simple_template = ChatPromptTemplate.from_template(
            "Bạn là trợ lý hỗ trợ nội bộ CT-Group. Trả lời rõ ràng, lịch sự:\n\nCâu hỏi: {question}"
        )

        cls.register(PromptType.RAG, rag_template)
        cls.register(PromptType.SIMPLE, simple_template)


# Khởi tạo mặc định khi import
PromptRegistry.register_defaults()