import time
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService

# ── Prompt cho Query Rewriting (cực ngắn để giảm latency) ──
_REWRITE_SYSTEM = """Bạn là bộ viết lại câu hỏi. Nhiệm vụ DUY NHẤT: đọc lịch sử hội thoại và câu hỏi follow-up, rồi viết lại thành MỘT câu hỏi ĐỘC LẬP rõ ràng bằng tiếng Việt.

Quy tắc:
- Chỉ trả về DUY NHẤT câu hỏi đã viết lại, không giải thích
- Giữ nguyên ý nghĩa gốc, chỉ bổ sung ngữ cảnh từ lịch sử
- Nếu câu hỏi đã rõ ràng (không phụ thuộc ngữ cảnh) → trả về nguyên văn"""


class RAGService:
    """RAG Service hoàn chỉnh với multi-turn conversation + Query Rewriting"""

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
        print(f"  ✏️  Query Rewrite                    : {t_rewrite:.2f}s")
        print(f"  📥 Retrieval (Embed+Search+Rerank) : {t_retrieval:.2f}s")
        print(f"  🤖 LLM Generation                  : {t_generation:.2f}s")
        print(f"  ──────────────────────────────────────────────")
        print(f"  🏁 TỔNG PIPELINE                    : {t_pipeline:.2f}s")
        print(f"{'='*60}\n")

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
        """Generator: Query Rewriting → Retrieval → Stream LLM output."""
        chat_history = chat_history or []

        t_total = time.time()

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
        print(f"  ✏️  Query Rewrite                    : {t_rewrite:.2f}s")
        print(f"  📥 Retrieval                        : {t_retrieval:.2f}s")
        print(f"  🤖 LLM Stream Generation            : {t_generation:.2f}s ({total_chars} chars)")
        print(f"  ──────────────────────────────────────────────")
        print(f"  🏁 TỔNG PIPELINE                    : {t_pipeline:.2f}s")
        print(f"{'='*60}\n")

