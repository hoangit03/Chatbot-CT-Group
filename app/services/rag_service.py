import re
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.services.retrieval import RetrievalService, RetrievalResult
from app.services.generation import GenerationService

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# Smart Router System Prompt  (hợp nhất: Intent + Rewrite + Keywords)
# ══════════════════════════════════════════════════════════════════════
_SMART_ROUTER_SYSTEM = """Bạn là bộ phân tích câu hỏi cho hệ thống tra cứu tài liệu nội bộ CT-Group.

NHIỆM VỤ: Phân tích câu hỏi của người dùng và trả về JSON DUY NHẤT (không giải thích).

Trả về chính xác JSON theo format:
{"intent": "...", "keywords": "...", "rewritten_query": "..."}

CÁC LOẠI INTENT:
- "spam": Câu vô nghĩa, không liên quan đến công việc, quy trình, chính sách CT-Group
  VD: "con mèo bay", "thời tiết hôm nay", "kể chuyện cười đi"
- "new_question": Câu hỏi MỚI, KHÔNG liên quan gì đến lịch sử hội thoại
  VD: "chính sách nghỉ phép", "quy trình tuyển dụng" (khi trước đó hỏi về lương)
- "follow_up": Câu hỏi TIẾP NỐI từ lịch sử hội thoại, cùng chủ đề
  VD: "chi tiết hơn được không?", "còn trường hợp nào khác?" (khi trước đó đã hỏi cùng chủ đề)

QUY TẮC KEYWORDS:
- Rút ra TỪ KHÓA CHÍNH liên quan đến quy trình/chính sách/quy định CT-Group
- Loại bỏ từ thừa: "tôi muốn hỏi", "bạn giúp tôi", "cho tôi biết", "là gì", "như thế nào"
- Giữ lại danh từ, động từ chính: "nghỉ phép", "lương thưởng", "thử việc", "BHXH"
- VD: "tôi muốn tìm hiểu về chính sách tiền lương" → keywords: "chính sách tiền lương"

QUY TẮC REWRITTEN_QUERY:
- Nếu intent=follow_up: viết lại câu hỏi thành câu ĐỘC LẬP, bổ sung ngữ cảnh từ lịch sử
- Nếu intent=new_question hoặc spam: copy nguyên keywords vào rewritten_query

CHỈ TRẢ VỀ JSON, KHÔNG giải thích.

/nothink"""


@dataclass
class SmartRouteResult:
    """Kết quả từ Smart Router."""
    intent: str           # "spam" | "new_question" | "follow_up"
    keywords: str         # Từ khóa chính rút gọn
    rewritten_query: str  # Câu hỏi đã viết lại (nếu follow_up)
    raw_query: str        # Câu hỏi gốc của user


# ══════════════════════════════════════════════════════════════════════
# Pre-RAG Regex Filters (Layer 1 — chạy TRƯỚC Smart Router)
# ══════════════════════════════════════════════════════════════════════
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

_INVALID_SHORT_RE = re.compile(
    r"^(\s*([a-z]{1,4}|\d{1,6}|[^\w\s]{2,}|string|test|asdf|qwer|zxcv|\.{2,}|\?{2,}|ok|okay|fine|no|yes)\s*)$",
    re.IGNORECASE,
)


def _is_gibberish(text: str) -> bool:
    """Phát hiện input vô nghĩa: ký tự lặp, spam, không có từ thật."""
    q = text.strip().lower()

    if len(q) < 3:
        return True
    if _INVALID_SHORT_RE.match(q):
        return True
    if len(q) <= 5 and not _CHITCHAT_RE.match(q):
        return True

    no_space = q.replace(" ", "")

    unique_ratio = len(set(no_space)) / max(len(no_space), 1)
    if len(no_space) > 4 and unique_ratio < 0.4:
        return True

    words = q.split()
    if any(len(w) > 12 for w in words):
        return True

    vietnamese_vowels = set("aeiouyàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ")
    # 4+ phụ âm liên tiếp TRONG 1 TỪ → không tồn tại trong tiếng Việt
    # Check từng từ riêng (KHÔNG check chuỗi nối liền vì ranh giới từ
    # tạo false positive: "đang"+"thắc" → "ngth" = 4 phụ âm)
    for word in words:
        consonant_streak = 0
        for c in word:
            if c.isalpha() and c not in vietnamese_vowels:
                consonant_streak += 1
                if consonant_streak >= 4:
                    return True
            else:
                consonant_streak = 0

    # Tất cả các từ đều không có nguyên âm → vô nghĩa
    has_any_vowel = any(c in vietnamese_vowels for w in words for c in w)
    if not has_any_vowel and len(q) > 3:
        return True

    return False


def _is_skip_rag(query: str) -> Optional[str]:
    """Layer 1: Regex filter — bypass cực nhanh (~0ms)."""
    q = query.strip()
    if _is_gibberish(q):
        return "gibberish"
    if _CHITCHAT_RE.match(q):
        return "chitchat"
    return None


# ══════════════════════════════════════════════════════════════════════
# RAG Service — Smart Router Architecture
# ══════════════════════════════════════════════════════════════════════

class RAGService:
    """RAG Service: Regex Filter → Smart Router → Retrieval → Generation"""

    def __init__(self):
        self.retrieval = RetrievalService()
        self.generation = GenerationService()

    def _smart_route(self, query: str, chat_history: List[BaseMessage]) -> SmartRouteResult:
        """
        🧠 Smart Router: 1 LLM call duy nhất thay thế Query Rewrite.
        
        Hợp nhất 3 chức năng:
          1. Intent Detection (spam / new_question / follow_up)
          2. Keyword Extraction (rút gọn query → keywords chính)
          3. Query Rewriting (nếu follow_up)

        Returns:
            SmartRouteResult với intent, keywords, rewritten_query
        """
        # Nếu không có history → luôn là new_question, chỉ cần extract keywords
        if not chat_history:
            # Gọi LLM để extract keywords (vẫn cần vì user hay viết dài)
            route_prompt = f"""Câu hỏi: {query}

(Không có lịch sử hội thoại)

Trả về JSON:"""
        else:
            # Build history context
            history_text = []
            for msg in chat_history[-4:]:
                if isinstance(msg, HumanMessage):
                    history_text.append(f"Người dùng: {msg.content[:200]}")
                elif isinstance(msg, AIMessage):
                    history_text.append(f"Bot: {msg.content[:150]}")  

            route_prompt = f"""Lịch sử hội thoại:
{chr(10).join(history_text)}

Câu hỏi hiện tại: {query}

Trả về JSON:"""

        messages = [
            SystemMessage(content=_SMART_ROUTER_SYSTEM),
            HumanMessage(content=route_prompt),
        ]

        # ── DEBUG: In input gửi vào Smart Router ──
        print(f"\n  {'─'*60}")
        print(f"  🧠 [SMART ROUTER] LLM Call #1 — Input:")
        print(f"  {'─'*60}")
        print(f"  📨 System prompt: {len(_SMART_ROUTER_SYSTEM)} ký tự")
        print(f"  📨 User prompt:")
        for line in route_prompt.strip().split('\n'):
            print(f"      {line}")
        print(f"  {'─'*60}")

        try:
            t0 = time.time()
            raw_response = self.generation.llm_client.invoke(messages).strip()
            t_elapsed = time.time() - t0

            # ── DEBUG: In raw output từ LLM ──
            print(f"\n  🧠 [SMART ROUTER] LLM Call #1 — Output ({t_elapsed:.2f}s):")
            print(f"  {'─'*60}")
            print(f"  📤 Raw response:")
            for line in raw_response.split('\n'):
                print(f"      {line}")
            print(f"  {'─'*60}")

            # Parse JSON từ response
            result = self._parse_router_json(raw_response, query)

            print(f"\n  🧠 [SMART ROUTER] Kết quả parse:")
            print(f"      Intent  : {result.intent}")
            print(f"      Keywords: \"{result.keywords}\"")
            if result.intent == "follow_up":
                print(f"      Rewrite : \"{result.rewritten_query}\"")

            return result

        except Exception as e:
            logger.warning(f"[Smart Router] Error: {e} → fallback to new_question")
            print(f"  ⚠️  [Smart Router] Lỗi: {e} → Fallback: new_question")
            return SmartRouteResult(
                intent="new_question",
                keywords=query,
                rewritten_query=query,
                raw_query=query,
            )

    def _parse_router_json(self, raw: str, original_query: str) -> SmartRouteResult:
        """Parse JSON output từ Smart Router, với fallback robust cho Qwen3."""
        # Strip <think>...</think> block nếu Qwen3 vẫn output thinking
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = raw  # Fallback nếu strip mất hết

        # Thử tìm JSON object trong response (hỗ trợ cả multiline)
        json_match = re.search(r'\{\s*"intent".*?\}', cleaned, re.DOTALL)
        if not json_match:
            # Fallback: tìm bất kỳ JSON-like block nào
            json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                intent = data.get("intent", "new_question")
                keywords = data.get("keywords", original_query)
                rewritten = data.get("rewritten_query", keywords)

                # Validate intent
                if intent not in ("spam", "new_question", "follow_up"):
                    intent = "new_question"

                # Validate keywords không rỗng
                if not keywords or len(keywords.strip()) < 2:
                    keywords = original_query

                # Validate rewritten_query hợp lệ
                if not rewritten or len(rewritten.strip()) < 3 or len(rewritten) > 500:
                    rewritten = keywords

                return SmartRouteResult(
                    intent=intent,
                    keywords=keywords.strip(),
                    rewritten_query=rewritten.strip(),
                    raw_query=original_query,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: không parse được JSON → coi là new_question, dùng query gốc
        logger.warning(f"[Smart Router] Cannot parse JSON from: {cleaned[:200]}")
        print(f"  ⚠️  [Smart Router] JSON parse fail → fallback: dùng query gốc")
        return SmartRouteResult(
            intent="new_question",
            keywords=original_query,
            rewritten_query=original_query,
            raw_query=original_query,
        )

    # ══════════════════════════════════════════════════════════════
    # answer() — Blocking mode
    # ══════════════════════════════════════════════════════════════

    def answer(self, query: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
        chat_history = chat_history or []

        t_total = time.time()
        t_route = 0.0
        t_retrieval = 0.0

        # ── Layer 1: Regex Filter (bypass cực nhanh) ──
        skip_reason = _is_skip_rag(query)

        if skip_reason:
            print(f"\n  ⚡ [Pre-RAG] Bypass RAG → Lý do: {skip_reason} | Query: \"{query}\"")
            retrieval_result = RetrievalResult(
                documents=[], query=query, top_k=0, total_retrieved=0, reranked=False
            )
        else:
            # ── Layer 2: Smart Router (1 LLM call) ──
            t_rw = time.time()
            route = self._smart_route(query, chat_history)
            t_route = time.time() - t_rw

            # Xử lý theo intent
            if route.intent == "spam":
                print(f"  🚫 [Smart Router] SPAM detected → Bypass RAG")
                retrieval_result = RetrievalResult(
                    documents=[], query=query, top_k=0, total_retrieved=0, reranked=False
                )
            else:
                # Chọn search query: keywords (new) hoặc rewritten (follow_up)
                if route.intent == "follow_up":
                    search_query = route.rewritten_query
                else:
                    search_query = route.keywords

                print(f"  🔍 [Search Query] \"{search_query}\"")

                # ── Layer 3: Retrieval ──
                t0 = time.time()
                retrieval_result = self.retrieval.retrieve(query=search_query)
                t_retrieval = time.time() - t0

                # Ghi đè query gốc cho Generation
                retrieval_result.query = query

        # ── Layer 4: Generation ──
        t1 = time.time()
        answer = self.generation.generate(retrieval_result, chat_history)
        t_generation = time.time() - t1

        t_pipeline = time.time() - t_total

        # ── Timing Report ──
        print(f"\n{'='*60}")
        print(f"  ⏱️  PIPELINE TIMING REPORT")
        print(f"{'='*60}")
        if skip_reason:
            print(f"  ⚡ Pre-RAG Bypass                    : {skip_reason}")
        else:
            print(f"  🧠 Smart Router                     : {t_route:.2f}s")
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

    # ══════════════════════════════════════════════════════════════
    # stream_answer() — Streaming mode
    # ══════════════════════════════════════════════════════════════

    def stream_answer(self, query: str, chat_history: List[BaseMessage] = None):
        """Generator: Regex Filter → Smart Router → Retrieval → Stream LLM.
        
        Yield 2 loại content:
        - Thinking steps: "<!-- thinking -->..." → UI hiện progress  
        - LLM content: text thường → UI hiện câu trả lời
        """
        chat_history = chat_history or []

        t_total = time.time()
        t_route = 0.0
        t_retrieval = 0.0

        # ── Layer 1: Regex Filter ──
        skip_reason = _is_skip_rag(query)

        if skip_reason:
            print(f"\n  ⚡ [Pre-RAG] Bypass RAG → Lý do: {skip_reason} | Query: \"{query}\"")
            retrieval_result = RetrievalResult(
                documents=[], query=query, top_k=0, total_retrieved=0, reranked=False
            )
        else:
            # ── Layer 2: Smart Router ──
            yield "<!-- thinking -->🧠 Đang phân tích câu hỏi...\n"
            t_rw = time.time()
            route = self._smart_route(query, chat_history)
            t_route = time.time() - t_rw

            if route.intent == "spam":
                print(f"  🚫 [Smart Router] SPAM → Bypass RAG")
                retrieval_result = RetrievalResult(
                    documents=[], query=query, top_k=0, total_retrieved=0, reranked=False
                )
            else:
                # search_query = keywords (new) hoặc rewritten_query (follow_up)
                search_query = route.rewritten_query if route.intent == "follow_up" else route.keywords
                print(f"  🔍 [Search Query] \"{search_query}\"")

                # ── Layer 3: Retrieval ──
                yield f"<!-- thinking -->🔍 Đang tìm kiếm tài liệu: \"{search_query[:50]}\"...\n"
                t0 = time.time()
                retrieval_result = self.retrieval.retrieve(query=search_query)
                t_retrieval = time.time() - t0
                print(f"  ⏱️  [Stream] Retrieval xong trong {t_retrieval:.2f}s")

                retrieval_result.query = query

                # Thông báo kết quả retrieval
                n_docs = len(retrieval_result.documents)
                if n_docs > 0:
                    yield f"<!-- thinking -->📚 Tìm thấy {n_docs} tài liệu liên quan\n"
                else:
                    yield "<!-- thinking -->📭 Không tìm thấy tài liệu phù hợp\n"

        # ── Layer 4: Stream LLM ──
        yield "<!-- thinking -->✍️ Đang soạn câu trả lời...\n"
        # Xóa thinking khi bắt đầu stream content thật
        yield "<!-- clear_thinking -->"

        t1 = time.time()
        total_chars = 0
        for chunk in self.generation.stream_generate(retrieval_result, chat_history):
            total_chars += len(chunk)
            yield chunk

        t_generation = time.time() - t1
        t_pipeline = time.time() - t_total

        # Timing Report
        print(f"\n{'='*60}")
        print(f"  ⏱️  STREAMING PIPELINE TIMING REPORT")
        print(f"{'='*60}")
        if skip_reason:
            print(f"  ⚡ Pre-RAG Bypass                    : {skip_reason}")
        else:
            print(f"  🧠 Smart Router                     : {t_route:.2f}s")
            print(f"  📥 Retrieval                        : {t_retrieval:.2f}s")
        print(f"  🤖 LLM Stream Generation            : {t_generation:.2f}s ({total_chars} chars)")
        print(f"  ──────────────────────────────────────────────")
        print(f"  🏁 TỔNG PIPELINE                    : {t_pipeline:.2f}s")
        print(f"{'='*60}\n")