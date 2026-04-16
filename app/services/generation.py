import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.utils.prompts import PromptRegistry, PromptType
from app.services.llm.ollama import BaseLLMClient, OllamaLLMClient
from app.services.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class GenerationService:
    """Generation service hỗ trợ multi-turn conversation"""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or OllamaLLMClient()
        logger.info("GenerationService đã khởi tạo")

    def generate(self, retrieval_result: RetrievalResult, chat_history: List[BaseMessage] = None) -> str:
        chat_history = chat_history or []

        if not retrieval_result.documents:
            prompt = PromptRegistry.get(PromptType.SIMPLE)
            messages = prompt.invoke({"question": retrieval_result.query}).messages
        else:
            context = "\n\n".join([
                f"Tài liệu: {doc.metadata.get('file_name', 'Unknown')}\n"
                f"Nội dung: {doc.page_content}"
                for doc in retrieval_result.documents
            ])
            prompt = PromptRegistry.get(PromptType.RAG)
            messages = prompt.invoke({
                "question": retrieval_result.query,
                "context": context
            }).messages

        # Kết hợp history + prompt hiện tại
        full_messages = chat_history + messages

        logger.info(f"[Generation] Đang sinh câu trả lời với history ({len(chat_history)} messages)")
        answer = self.llm_client.invoke(full_messages)

        return answer