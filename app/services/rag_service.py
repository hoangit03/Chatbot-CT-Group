import re
import time
from typing import List, Dict, Any, Optional, AsyncIterator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.services.retrieval import RetrievalService, RetrievalResult
from app.services.generation import GenerationService
import logging

logger = logging.getLogger(__name__)

# ── Prompt cho Query Rewriting (cực ngắn để giảm latency) ──
_REWRITE_SYSTEM = (
    "Bạn là bộ viết lại câu hỏi. Nhiệm vụ DUY NHẤT: đọc lịch sử hội thoại "
    "và câu hỏi follow-up, rồi viết lại thành MỘT câu hỏi ĐỘC LẬP rõ ràng "
    "bằng tiếng Việt.\n\nQuy tắc:\n"
    "- Chỉ trả về DUY NHẤT câu hỏi đã viết lại, không giải thích\n"
    "- Giữ nguyên ý nghĩa gốc, chỉ bổ sung ngữ cảnh từ lịch sử\n"
    "- Nếu câu hỏi đã rõ ràng → trả về nguyên văn"
)

# ── Pre-RAG Intent Classification ──
# Những câu này bypass hoàn toàn VectorDB + Reranker + Query Rewrite
_CHITCHAT_RE = re.compile(
    r"""
    ^(\s*(
        xin\s*chào | chào\s*(buổi)?\s*(sáng|trưa|chiều|tối|anh|chị|bạn|em|mn|mọi\s*người)?
        | hé\s*l[oô] | hell+o+ | hi+\s*$| hey+ | howdy | good\s*(morning|afternoon|evening|night)
        | alo+ | ơi | ê\s*$
        | bạn\s*(là\s*(ai|gì)|làm\s*(được\s*)?gì|có\s*thể\s*giúp|giúp\s*được\s*gì|có\s*khỏe|khỏe\s*không|khỏe\s*hong)
        | chatbot\s*(này\s*)?(là\s*gì|dùng\s*để|làm\s*gì)
        | (em|bạn)\s*(là\s*ai|là\s*gì|tên\s*gì|ơi)
        | introduce\s*yourself | who\s*are\s*you | what\s*can\s*you\s*do
        | (cảm?\s*ơn|thanks?|thank\s*you|tks|ty|camon|cam\s*on)(.{0,40})?
        | bạn\s*giỏi\s*(quá|thật|vậy) | (hay|tốt|được)\s*(lắm|quá|đấy|vậy)
        | (tạm\s*biệt|bye+|goodbye|gặp\s*lại|hẹn\s*gặp|see\s*you)(.{0,40})?
        | mấy\s*giờ\s*(rồi)? | bây\s*giờ\s*là\s*mấy\s*giờ | what\s*time
        | hôm\s*nay\s*(là\s*)?(ngày|thứ)\s*mấy | today\s*is
        | (tôi|mình)\s*cần\s*(bạn\s*)?(giúp|hỗ\s*trợ)\s*(tôi\s*)?
        | giúp\s*(tôi|mình)\s*(với|đi|nha|nhé)?
    )\s*)$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Regex cho invalid/gibberish đơn giản  
_INVALID_SHORT_RE = re.compile(
    r"^(\s*([a-z]{1,4}|\d{1,6}|[^\w\s]{2,}|string|test|asdf|qwer|zxcv|\.{2,}|\?{2,}|ok|okay|fine|no|yes)\s*)$",
    re.IGNORECASE,
)

_VIETNAMESE_VOWELS = frozenset(
    "aeiouyàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
)


def _is_gibberish(text: str) -> bool:
    """Phát hiện input vô nghĩa: ký tự lặp, spam, không có từ thật."""
    q = text.strip().lower()
    
    # Quá ngắn (< 3 ký tự)
    if len(q) < 3:
        return True
    
    # Invalid patterns cơ bản (test, asdf, ok, ...)
    if _INVALID_SHORT_RE.match(q):
        return True
    
    # Chuỗi ngắn ≤ 5 ký tự mà không phải pattern đã biết → gibberish
    # (từ tiếng Việt có nghĩa thường đi kèm dấu hoặc ngữ cảnh dài hơn)
    if len(q) <= 5 and not _CHITCHAT_RE.match(q):
        return True
    
    no_space = q.replace(" ", "")
    
    # Quá ít ký tự unique (vd: "gugugu", "aaaaa")
    unique_ratio = len(set(no_space)) / max(len(no_space), 1)
    if len(no_space) > 4 and unique_ratio < 0.4:
        return True
    
    # Bất kỳ "từ" nào dài > 12 ký tự → rất khó là TV/mã nội bộ hợp lệ
    # (TV max ~7 ký tự, mã nội bộ ~10: "BNLCĐ-QĐ201")
    words = q.split()
    if any(len(w) > 12 for w in words):
        return True
    
    # 4+ phụ âm liên tiếp → không tồn tại trong tiếng Việt
    consonant_streak = 0
    for c in no_space:
        if c.isalpha() and c not in _VIETNAMESE_VOWELS:
            consonant_streak += 1
            if consonant_streak >= 4:
                return True
        else:
            consonant_streak = 0
    
    # Không chứa nguyên âm nào → vô nghĩa
    if not any(c in _VIETNAMESE_VOWELS for c in q) and len(q) > 3:
        return True
    
    return False

def _classify_query(query: str) -> Optional[str]:
    """Returns 'chitchat' | 'gibberish' | None. None = chạy full RAG."""
    q = query.strip()
    if _is_gibberish(q):
        return "gibberish"
    if _CHITCHAT_RE.match(q):
        return "chitchat"
    return None
 
 
def _empty_retrieval(query: str) -> RetrievalResult:
    return RetrievalResult(documents=[], query=query, top_k=0, total_retrieved=0, reranked=False)


class RAGService:
    """RAG Service: Pre-RAG Intent Classification + Query Rewriting + RAG Pipeline"""

    def __init__(self):
        self.retrieval = RetrievalService()
        self.generation = GenerationService()

    def _rewrite_query(self, query: str, chat_history: List[BaseMessage]) -> str:
        """
        Viết lại câu hỏi follow-up thành câu hỏi độc lập.
        
        Ví dụ:
          History: "thử việc thành công" → answer...
          Follow-up: "chi tiết hơn được không?"
          Rewritten: "hướng dẫn chi tiết cách thử việc thành công tại CT Group"
        """
        # Nếu không có history → không cần rewrite
        if not chat_history:
            return query

        messages = self._build_rewrite_messages(query, chat_history)
        try:
            rewritten = self.generation.llm_client.invoke(messages).strip()
            return rewritten if 3 < len(rewritten) < 500 else query
        except Exception as exc:
            logger.warning("[QueryRewrite] sync error, using original: %s", exc)
            return query

    async def _arewrite_query(self, query: str, chat_history: List[BaseMessage]) -> str:
        """FIX [BUG-3]: async version dùng ainvoke, không block event loop."""
        if not chat_history:
            return query
        messages = self._build_rewrite_messages(query, chat_history)
        try:
            rewritten = (await self.generation.llm_client.ainvoke(messages)).strip()
            return rewritten if 3 < len(rewritten) < 500 else query
        except Exception as exc:
            logger.warning("[QueryRewrite] async error, using original: %s", exc)
            return query

    def _build_rewrite_messages(
        self, query: str, chat_history: List[BaseMessage]
    ) -> List[BaseMessage]:
        history_lines = []
        for msg in chat_history[-4:]:
            if isinstance(msg, HumanMessage):
                history_lines.append(f"Người dùng: {msg.content[:200]}")
            elif isinstance(msg, AIMessage):
                history_lines.append(f"Bot: {msg.content[:200]}")
        content = (
            f"Lịch sử hội thoại:\n{chr(10).join(history_lines)}\n\n"
            f"Câu hỏi follow-up: {query}\n\n"
            "Viết lại thành câu hỏi độc lập:"
        )
        return [SystemMessage(content=_REWRITE_SYSTEM), HumanMessage(content=content)]

    async def _do_retrieval_async(
        self, query: str, chat_history: List[BaseMessage]
    ) -> RetrievalResult:
        """Rewrite + retrieve (async). Tái sử dụng cho aanswer + astream_answer."""
        search_query = await self._arewrite_query(query, chat_history)
        if search_query != query:
            logger.debug("[QueryRewrite] %r → %r", query, search_query)
        result = await self.retrieval.aretrieve(query=search_query)
        result.query = query  # restore original query cho generation
        return result
 
    def _do_retrieval_sync(
        self, query: str, chat_history: List[BaseMessage]
    ) -> RetrievalResult:
        search_query = self._rewrite_query(query, chat_history)
        result = self.retrieval.retrieve(query=search_query)
        result.query = query
        return result

    def answer(self, query: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
        """"""
        chat_history = chat_history or []
        t0 = time.perf_counter()
 
        skip_reason = _classify_query(query)
        if skip_reason:
            logger.debug("[PreRAG] bypass=%s query=%r", skip_reason, query)
            retrieval_result = _empty_retrieval(query)
        else:
            retrieval_result = self._do_retrieval_sync(query, chat_history)
 
        answer = self.generation.generate(retrieval_result, chat_history)
        logger.info("[answer] total=%.2fs docs=%d", time.perf_counter() - t0, len(retrieval_result.documents))
        return self._build_response(query, answer, retrieval_result)

    @staticmethod
    def _build_response(query, answer, retrieval_result) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "file_name": doc.metadata.get("file_name"),
                    "score": doc.metadata.get("rerank_score")
                    or doc.metadata.get("similarity_score", 0),
                }
                for doc in retrieval_result.documents
            ],
            "retrieved_count": len(retrieval_result.documents)
        }

    def stream_answer(self, query: str, chat_history: List[BaseMessage] = None):
        """Generator: Pre-RAG Classification → Query Rewriting → Retrieval → Stream LLM."""
        chat_history = chat_history or []
 
        skip_reason = _classify_query(query)
        if skip_reason:
            retrieval_result = _empty_retrieval(query)
        else:
            retrieval_result = self._do_retrieval_sync(query, chat_history)
 
        yield from self.generation.stream_generate(retrieval_result, chat_history)

    async def astream_answer(
        self, query: str, chat_history: Optional[List[BaseMessage]] = None
    ) -> AsyncIterator[str]:
        """
        FIX [BUG-4] + [PERF-1]: native async generator.
        Dùng cho async def endpoint — không chiếm thread pool.
        """
        chat_history = chat_history or []
 
        skip_reason = _classify_query(query)
        if skip_reason:
            logger.debug("[PreRAG] bypass=%s", skip_reason)
            retrieval_result = _empty_retrieval(query)
        else:
            retrieval_result = await self._do_retrieval_async(query, chat_history)
 
        async for chunk in self.generation.astream_generate(retrieval_result, chat_history):
            yield chunk

    async def aanswer(
        self, query: str, chat_history: List[BaseMessage] = None
    ) -> Dict[str, Any]:
        """
        """
        chat_history = chat_history or []
 
        retrieval_result = await self.retrieval.aretrieve(query=query)
        answer = await self.generation.agenerate(retrieval_result, chat_history)
 
        return self._build_response(query, answer, retrieval_result)
