import logging
import re
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.utils.prompts import PromptRegistry, PromptType
from app.services.llm.ollama import BaseLLMClient, OllamaLLMClient
from app.services.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

_CHITCHAT_PATTERNS = re.compile(
    r"""
    ^(\s*(
        # Chào hỏi
        xin\s*chào | chào\s*(buổi)?\s*(sáng|trưa|chiều|tối|anh|chị|bạn|em|mn|mọi\s*người)?
        | hello | hi+| hey | howdy | good\s*(morning|afternoon|evening|night)
        | alo | ơi
 
        # Hỏi về bot
        | bạn\s*(là\s*(ai|gì)|làm\s*(được\s*)?gì|có\s*thể\s*giúp|giúp\s*được\s*gì|dùng\s*để\s*làm\s*gì)
        | chatbot\s*(này\s*)?(là\s*gì|dùng\s*để|làm\s*gì|hoạt\s*động)
        | (cho\s*mình?\s*)?biết\s*(thêm\s*)?về\s*(bot|chatbot|trợ\s*lý)
        | mày\s*(là\s*ai|làm\s*gì)
        | (em|bạn)\s*(là\s*ai|là\s*gì|tên\s*gì)
        | introduce\s*yourself | who\s*are\s*you | what\s*can\s*you\s*do
 
        # Cảm ơn
        | (cảm?\s*ơn|thanks?|thank\s*you|tks|ty|camon|cam\s*on)(.*)?
        | bạn\s*giỏi\s*(quá|thật|vậy)
        | (hay|tốt|được)\s*(lắm|quá|đấy|vậy)
 
        # Tạm biệt
        | (tạm\s*biệt|bye+|goodbye|gặp\s*lại|hẹn\s*gặp|see\s*you)(.*)?
 
        # Vô nghĩa / test
        | [a-z]{1,3}                    # chuỗi rất ngắn kiểu "ok", "ah"
        | [^\w\s]{2,}                   # chỉ toàn ký tự đặc biệt
        | \d{1,6}                       
 
    )\s*)$
    """,
    re.VERBOSE | re.IGNORECASE,
)

_INVALID_QUERY_PATTERNS = re.compile(
    r"^(\s*("
    r"[a-z]{1,4}|"                    
    r"\d{1,6}|"                      
    r"[^\w\s]{2,}|"                   
    r"string|test|asdf|qwer|zxcv|"
    r"\.{2,}|"                        
    r"\?{2,}|"                        
    r"ok|okay|fine|no|yes"
    r")\s*)$",
    re.IGNORECASE | re.VERBOSE,
)


def _is_invalid_query(question: str) -> bool:
    """Kiểm tra query vô nghĩa, test, rác"""
    q = question.strip()
    if len(q) < 4:                                      # quá ngắn
        return True
    if _INVALID_QUERY_PATTERNS.match(q):                # khớp regex rác
        return True
    if len(set(q.lower())) < 3 and len(q) > 8:         # lặp ký tự vô nghĩa
        return True
    return False

def _is_chitchat(question: str) -> bool:
    """Kiểm tra nhanh bằng regex trước khi quyết định dùng prompt nào."""
    return bool(_CHITCHAT_PATTERNS.match(question.strip()))

class GenerationService:
    """Generation service hỗ trợ multi-turn conversation"""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or OllamaLLMClient()
        logger.info("GenerationService đã khởi tạo")

    def generate(self, retrieval_result: RetrievalResult, chat_history: List[BaseMessage] = None) -> str:
        chat_history = chat_history or []
        question = retrieval_result.query

        if _is_invalid_query(question):
            prompt_type = PromptType.INVALID_QUERY
            prompt_vars = {"question": question, "chat_history": chat_history}
            logger.info("[Generation] Intent: INVALID_QUERY")

        if _is_chitchat(question):
            prompt_type = PromptType.CHITCHAT
            prompt_vars = {"question": question, "chat_history": chat_history}
            logger.info("[Generation] Intent: CHITCHAT")
        
        if not retrieval_result.documents:
            prompt_type = PromptType.SIMPLE
            prompt_vars = {"question": question, "chat_history": chat_history}
            logger.info("[Generation] Intent: SIMPLE (không tìm thấy tài liệu)")
        else:
            # === DYNAMIC CONTEXT ASSEMBLER ===
            # Giới hạn dung lượng text được nhồi vào LLM để tối ưu Tốc Độ & Chống Quá Tải
            # Ví dụ: 2500 ký tự (Khoảng 500-600 tokens) là mức lý tưởng cho Model Qwen 1.7B
            max_char_limit = 2500
            current_chars = 0
            selected_docs = []
            
            for doc in retrieval_result.documents:
                doc_len = len(doc.page_content)
                # Đảm bảo Bot luôn có ÍT NHẤT 1 chunk dù chunk đó có to đến mấy
                if current_chars + doc_len > max_char_limit and len(selected_docs) > 0:
                    logger.info(f"[Generation] Đã chạm ngưỡng {current_chars}/{max_char_limit} chars. Tự động ngắt nạp thêm Chunk.")
                    break
                    
                selected_docs.append(doc)
                current_chars += doc_len

            context = "\n\n".join([
                f"Tài liệu: {doc.metadata.get('source', doc.metadata.get('file_name', 'Unknown'))}\n"
                f"Nội dung: {doc.page_content}"
                for doc in selected_docs
            ])
            
            prompt_type = PromptType.RAG
            prompt_vars = {
                "question": question,
                "context": context,
                "chat_history": chat_history,
            }
            logger.info(
                f"[Generation] Intent: RAG | Đã dùng {len(selected_docs)}/{len(retrieval_result.documents)} Chunks (Tổng: {current_chars} ký tự)"
            )

        # Kết hợp history + prompt hiện tại
        prompt = PromptRegistry.get(prompt_type)
        messages = prompt.invoke(prompt_vars).messages

        logger.info(f"[Generation] Đang sinh câu trả lời với history ({len(chat_history)} messages)")
        answer = self.llm_client.invoke(messages)

        return answer