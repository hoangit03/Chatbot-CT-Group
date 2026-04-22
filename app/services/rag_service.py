import re
import time
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.services.retrieval import RetrievalService, RetrievalResult
from app.services.generation import GenerationService

# ── Prompt cho Query Rewriting (cực ngắn để giảm latency) ──
_REWRITE_SYSTEM = """Bạn là bộ viết lại câu hỏi. Nhiệm vụ DUY NHẤT: đọc lịch sử hội thoại và câu hỏi follow-up, rồi viết lại thành MỘT câu hỏi ĐỘC LẬP rõ ràng bằng tiếng Việt.

Quy tắc:
- Chỉ trả về DUY NHẤT câu hỏi đã viết lại, không giải thích
- Giữ nguyên ý nghĩa gốc, chỉ bổ sung ngữ cảnh từ lịch sử
- Nếu câu hỏi đã rõ ràng (không phụ thuộc ngữ cảnh) → trả về nguyên văn"""

# ── Pre-RAG Intent Classification ──
# Những câu này bypass hoàn toàn VectorDB + Reranker + Query Rewrite
_CHITCHAT_RE = re.compile(
    r"""
    ^(\s*(
        xin\s*chào | chào\s*(buổi)?\s*(sáng|trưa|chiều|tối|anh|chị|bạn|em|mn|mọi\s*người)?
        | hello | hé\s*lô | hi+ | hey | howdy | good\s*(morning|afternoon|evening|night)
        | alo | ơi
        | bạn\s*(là\s*(ai|gì)|làm\s*(được\s*)?gì|có\s*thể\s*giúp|giúp\s*được\s*gì|có\s*khỏe|khỏe\s*không|khỏe\s*hong)
        | chatbot\s*(này\s*)?(là\s*gì|dùng\s*để|làm\s*gì)
        | (em|bạn)\s*(là\s*ai|là\s*gì|tên\s*gì)
        | introduce\s*yourself | who\s*are\s*you | what\s*can\s*you\s*do
        | (cảm?\s*ơn|thanks?|thank\s*you|tks|ty|camon|cam\s*on)(.{0,40})?
        | bạn\s*giỏi\s*(quá|thật|vậy) | (hay|tốt|được)\s*(lắm|quá|đấy|vậy)
        | (tạm\s*biệt|bye+|goodbye|gặp\s*lại|hẹn\s*gặp|see\s*you)(.{0,40})?
    )\s*)$
    """,
    re.VERBOSE | re.IGNORECASE,
)

_VAGUE_RE = re.compile(
    r"""
    ^(\s*(
        tôi\s*cần\s*(bạn\s*)?(giúp|hỗ\s*trợ)\s*(tôi\s*)?$
        | giúp\s*(tôi|mình)\s*(với|đi|nha|nhé)?\s*$
        | (bạn|em)\s*có\s*khỏe\s*(hong|không|k)\s*$
        | hỏi\s*(cái|1|một)\s*(gì|chút)\s*$
        | có\s*ai\s*(đó|ở\s*đây)?\s*$
    )\s*)$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _is_skip_rag(query: str) -> str:
    """
    Phân loại TRƯỚC khi vào RAG pipeline.
    
    Returns:
        "chitchat" | "vague" | None
        - chitchat: chào hỏi, cảm ơn, tạm biệt → bypass RAG
        - vague: câu quá mơ hồ → bypass RAG, yêu cầu cụ thể hơn
        - None: câu hỏi thật → chạy full RAG pipeline
    """
    q = query.strip()
    if len(q) < 3:
        return "vague"
    if _CHITCHAT_RE.match(q):
        return "chitchat"
    if _VAGUE_RE.match(q):
        return "vague"
    return None


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

        # Build conversation context cho LLM rewriter
        history_text = []
        for msg in chat_history[-4:]:  # Chỉ lấy 2 cặp Q&A gần nhất (tiết kiệm token)
            if isinstance(msg, HumanMessage):
                history_text.append(f"Người dùng: {msg.content[:200]}")
            elif isinstance(msg, AIMessage):
                history_text.append(f"Bot: {msg.content[:200]}")

        rewrite_prompt = f"""Lịch sử hội thoại:
{chr(10).join(history_text)}

Câu hỏi follow-up: {query}

Viết lại thành câu hỏi độc lập:"""

        messages = [
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(content=rewrite_prompt),
        ]

        try:
            rewritten = self.generation.llm_client.invoke(messages).strip()
            # Đảm bảo kết quả hợp lệ
            if rewritten and len(rewritten) > 3 and len(rewritten) < 500:
                return rewritten
            return query
        except Exception as e:
            print(f"  ⚠️  [Query Rewrite] Lỗi: {e} → Dùng query gốc")
            return query

    def answer(self, query: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
        chat_history = chat_history or []

        t_total = time.time()
        t_rewrite = 0.0
        t_retrieval = 0.0

        # ── Bước -1: Pre-RAG Intent Classification ──
        skip_reason = _is_skip_rag(query)

        if skip_reason:
            print(f"\n  ⚡ [Pre-RAG] Bypass RAG → Lý do: {skip_reason} | Query: \"{query}\"")
            # Tạo RetrievalResult rỗng, Generation sẽ dùng CHITCHAT/SIMPLE prompt
            retrieval_result = RetrievalResult(documents=[], query=query, top_k=0, reranked=False)
        else:
            # ── Bước 0: Query Rewriting (nếu có chat_history) ──
            t_rw = time.time()
            search_query = self._rewrite_query(query, chat_history)
            t_rewrite = time.time() - t_rw

            if search_query != query:
                print(f"\n  ✏️  [Query Rewrite] {t_rewrite:.2f}s")
                print(f"      Gốc     : \"{query}\"")
                print(f"      Viết lại: \"{search_query}\"")
            else:
                print(f"\n  ✏️  [Query Rewrite] Không cần viết lại (query đã rõ ràng)")

            # ── Bước 1: Retrieval (dùng search_query đã rewrite) ──
            t0 = time.time()
            retrieval_result = self.retrieval.retrieve(query=search_query)
            t_retrieval = time.time() - t0

            # Ghi đè query gốc lại để Generation nhận đúng câu hỏi người dùng
            retrieval_result.query = query

        # ── Bước 2: Generation (LLM Inference) ──
        t1 = time.time()
        answer = self.generation.generate(retrieval_result, chat_history)
        t_generation = time.time() - t1

        t_pipeline = time.time() - t_total

        # ── Debug Timing Report ──
        print(f"\n{'='*60}")
        print(f"  ⏱️  PIPELINE TIMING REPORT")
        print(f"{'='*60}")
        if skip_reason:
            print(f"  ⚡ Pre-RAG Bypass                    : {skip_reason}")
        else:
            print(f"  ✏️  Query Rewrite                    : {t_rewrite:.2f}s")
            print(f"  📥 Retrieval (Embed+Search+Rerank) : {t_retrieval:.2f}s")
        print(f"  🤖 LLM Generation                  : {t_generation:.2f}s")
        print(f"  ──────────────────────────────────────────────")
        print(f"  🏁 TỔNG PIPELINE                    : {t_pipeline:.2f}s")
        print(f"{'='*60}\n")

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "file_name": doc.metadata.get("source_file") or doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown",
                    "score": doc.metadata.get("rerank_score") or doc.metadata.get("similarity_score", 0)
                }
                for doc in retrieval_result.documents
            ],
            "retrieved_count": len(retrieval_result.documents)
        }

    def stream_answer(self, query: str, chat_history: List[BaseMessage] = None):
        """Generator: Pre-RAG Classification → Query Rewriting → Retrieval → Stream LLM."""
        chat_history = chat_history or []

        t_total = time.time()
        t_rewrite = 0.0
        t_retrieval = 0.0

        # ── Bước -1: Pre-RAG Intent Classification ──
        skip_reason = _is_skip_rag(query)

        if skip_reason:
            print(f"\n  ⚡ [Pre-RAG] Bypass RAG → Lý do: {skip_reason} | Query: \"{query}\"")
            retrieval_result = RetrievalResult(documents=[], query=query, top_k=0, reranked=False)
        else:
            # Bước 0: Query Rewriting
            t_rw = time.time()
            search_query = self._rewrite_query(query, chat_history)
            t_rewrite = time.time() - t_rw

            if search_query != query:
                print(f"\n  ✏️  [Query Rewrite] {t_rewrite:.2f}s")
                print(f"      Gốc     : \"{query}\"")
                print(f"      Viết lại: \"{search_query}\"")
            else:
                print(f"\n  ✏️  [Query Rewrite] Không cần viết lại")

            # Bước 1: Retrieval (dùng search_query đã rewrite)
            t0 = time.time()
            retrieval_result = self.retrieval.retrieve(query=search_query)
            t_retrieval = time.time() - t0
            print(f"  ⏱️  [Stream] Retrieval xong trong {t_retrieval:.2f}s")

            # Ghi đè query gốc lại cho Generation
            retrieval_result.query = query

        # Bước 2: Stream LLM Generation
        t1 = time.time()
        total_chars = 0
        for chunk in self.generation.stream_generate(retrieval_result, chat_history):
            total_chars += len(chunk)
            yield chunk

        t_generation = time.time() - t1
        t_pipeline = time.time() - t_total

        # Debug Timing Report
        print(f"\n{'='*60}")
        print(f"  ⏱️  STREAMING PIPELINE TIMING REPORT")
        print(f"{'='*60}")
        if skip_reason:
            print(f"  ⚡ Pre-RAG Bypass                    : {skip_reason}")
        else:
            print(f"  ✏️  Query Rewrite                    : {t_rewrite:.2f}s")
            print(f"  📥 Retrieval                        : {t_retrieval:.2f}s")
        print(f"  🤖 LLM Stream Generation            : {t_generation:.2f}s ({total_chars} chars)")
        print(f"  ──────────────────────────────────────────────")
        print(f"  🏁 TỔNG PIPELINE                    : {t_pipeline:.2f}s")
        print(f"{'='*60}\n")