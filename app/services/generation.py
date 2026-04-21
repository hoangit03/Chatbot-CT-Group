import logging
import re
import unicodedata
import base64
from typing import List, Optional

from langchain_core.messages import BaseMessage

from app.utils.prompts import PromptRegistry, PromptType
from app.services.llm.ollama import BaseLLMClient, OllamaLLMClient
from app.services.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

# =============================================================================
# LAYER 1A — INPUT SANITIZER
# Chống: Encoding Attacks, Delimiter Confusion, Token Smuggling
# =============================================================================

# Tags nguy hiểm có thể phá vỡ cấu trúc prompt
_DANGEROUS_TAGS = re.compile(
    r'===\s*(BEGIN|END)_DOC\s*===|'
    r'<\s*/?\s*(system|instruction|context|prompt|human|assistant)'
    r'[^>]*>',
    re.IGNORECASE,
)

# Patterns injection cố gắng đặt lệnh trong context/metadata
_INJECTION_PATTERNS = re.compile(
    r'(ignore|forget|disregard|bypass|override|dismiss|circumvent)'
    r'.{0,30}(instruction|rule|prompt|system|previous|above|prior)',
    re.IGNORECASE,
)

def _decode_and_normalize(text: str) -> str:
    """
    Chống Encoding Attacks:
    - Decode base64 nếu toàn bộ text là base64
    - Normalize unicode (NFC) để phát hiện zero-width chars, homoglyphs
    - Remove zero-width spaces và control characters
    """
    # Thử decode base64
    try:
        stripped = text.strip()
        if len(stripped) > 8 and len(stripped) % 4 == 0:
            decoded = base64.b64decode(stripped, validate=True).decode('utf-8', errors='ignore')
            if decoded.isprintable() or '\n' in decoded:
                logger.warning("[Security] Base64 encoded input detected and decoded")
                text = decoded
    except Exception:
        pass

    # Normalize unicode — phát hiện homoglyph và zero-width attacks
    text = unicodedata.normalize('NFC', text)

    # Remove zero-width characters thường dùng để bypass regex
    zero_width = ['\u200b', '\u200c', '\u200d', '\u200e', '\u200f',
                  '\ufeff', '\u2060', '\u2061', '\u2062', '\u2063']
    for zw in zero_width:
        text = text.replace(zw, '')

    # Remove non-printable control characters (trừ newline/tab)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C'
                   or c in ('\n', '\t', '\r'))

    return text.strip()


def _sanitize_document_content(content: str, file_name: str) -> str:
    """
    Chống Context Poisoning & Delimiter Confusion:
    - Xóa các tag cố tình phá cấu trúc prompt
    - Đánh dấu các đoạn có injection pattern để LLM biết đây là dữ liệu đáng ngờ
    - Wrap bằng delimiter duy nhất, khó giả mạo
    """
    # Xóa các dangerous tags
    content = _DANGEROUS_TAGS.sub('[TAG_REMOVED]', content)

    # Kiểm tra injection pattern trong document
    if _INJECTION_PATTERNS.search(content):
        logger.warning(f"[Security] Possible context poisoning in doc: {file_name}")
        content = _INJECTION_PATTERNS.sub('[CONTENT_FLAGGED]', content)

    # Giới hạn độ dài để tránh token flooding
    max_content_length = 4000
    if len(content) > max_content_length:
        content = content[:max_content_length] + '\n[... nội dung bị cắt bớt ...]'

    return content


def _sanitize_metadata(value: str, max_length: int = 200) -> str:
    """Sanitize metadata fields (file_name, etc.) — chống metadata injection"""
    if not isinstance(value, str):
        value = str(value)
    # Xóa newlines và tags trong metadata
    value = re.sub(r'[\n\r]', ' ', value)
    value = _DANGEROUS_TAGS.sub('[REMOVED]', value)
    return value[:max_length]


def _build_safe_context(documents) -> str:
    """
    Build context với delimiter rõ ràng, khó giả mạo.
    Dùng ===BEGIN_DOC=== / ===END_DOC=== thay vì XML tags
    để tránh LLM nhầm với system instructions.
    """
    safe_parts = []
    for i, doc in enumerate(documents):
        file_name = _sanitize_metadata(
            doc.metadata.get('file_name', 'Không rõ nguồn')
        )
        content = _sanitize_document_content(doc.page_content, file_name)

        safe_parts.append(
            f"===BEGIN_DOC id={i+1} source=\"{file_name}\"===\n"
            f"{content}\n"
            f"===END_DOC id={i+1}==="
        )

    return "\n\n".join(safe_parts)


# =============================================================================
# LAYER 1B — INTENT DETECTOR
# Chống: Typo/Synonym Attack, DAN-style, Refusal Suppression, Few-shot Injection
# =============================================================================

# Jailbreak patterns — bắt cả typo, synonym, obfuscation phổ biến
_JAILBREAK_PATTERNS = re.compile(
    r"""
    # Classic ignore/forget patterns (với typo phổ biến)
    \b(ign[o0]re|forg[e3]t|disregard|dismiss|bypass|circumvent|overr?ide|reset)\b
    .{0,40}
    \b(instruct|rule|prompt|system|previous|above|prior|restrict)\b

    # Role override
    | \b(you\s*are\s*(now|a|an)|act\s*as|pretend\s*(to\s*be|you\s*are)|
         roleplay\s*as|simulate|imagine\s*you\s*are|behave\s*as)\b
    .{0,50}
    \b(without\s*restrict|no\s*limit|no\s*filter|no\s*rule|
       uncensored|unrestricted|free\s*ai|jailbreak)\b

    # DAN-style
    | \b(DAN|do\s*anything\s*now|developer\s*mode|god\s*mode|
         jailbreak\s*mode|unlock\s*mode|unrestricted\s*mode)\b

    # Refusal suppression
    | \b(never\s*(refuse|decline|say\s*no)|always\s*(answer|comply|agree)|
         must\s*(answer|respond|comply)|cannot\s*(refuse|decline))\b

    # System prompt leak attempts
    | (repeat|print|show|display|reveal|tell\s*me|what\s*(is|are))
    .{0,30}
    (system\s*prompt|instruct|your\s*rules?|your\s*guidelines?)

    # Fictional framing bypass
    | (hypothetically|in\s*a\s*story|for\s*fiction|in\s*a\s*(movie|game|book)|
       my\s*grandmother\s*(used\s*to|would)|pretend\s*this\s*is)
    .{0,60}
    (ignore|bypass|no\s*rule|unrestrict)

    # Few-shot injection markers
    | (human\s*:|user\s*:|assistant\s*:|bot\s*:)\s*(ignore|bypass|you\s*are\s*now)

    # Tiếng Việt
    | (bỏ\s*qua|quên|không\s*cần\s*theo)\s*(quy\s*tắc|hướng\s*dẫn|lệnh|ràng\s*buộc)
    | (bạn\s*là|bây\s*giờ\s*bạn\s*là|đóng\s*vai)\s*.{0,50}(không\s*có\s*giới\s*hạn|tự\s*do)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CHITCHAT_PATTERNS = re.compile(
    r"""
    ^(\s*(
        xin\s*chào | chào\s*(buổi)?\s*(sáng|trưa|chiều|tối|anh|chị|bạn|em|mn|mọi\s*người)?
        | hello | hi+ | hey | howdy | good\s*(morning|afternoon|evening|night)
        | alo | ơi

        | bạn\s*(là\s*(ai|gì)|làm\s*(được\s*)?gì|có\s*thể\s*giúp|giúp\s*được\s*gì)
        | chatbot\s*(này\s*)?(là\s*gì|dùng\s*để|làm\s*gì)
        | (em|bạn)\s*(là\s*ai|là\s*gì|tên\s*gì)
        | introduce\s*yourself | who\s*are\s*you | what\s*can\s*you\s*do

        | (cảm?\s*ơn|thanks?|thank\s*you|tks|ty|camon|cam\s*on)(.{0,40})?
        | bạn\s*giỏi\s*(quá|thật|vậy)
        | (hay|tốt|được)\s*(lắm|quá|đấy|vậy)

        | (tạm\s*biệt|bye+|goodbye|gặp\s*lại|hẹn\s*gặp|see\s*you)(.{0,40})?
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
    r"\.{2,}|\?{2,}|"
    r"ok|okay|fine|no|yes"
    r")\s*)$",
    re.IGNORECASE,
)


def _is_invalid_query(question: str) -> bool:
    q = question.strip()
    if len(q) < 4:
        return True
    if _INVALID_QUERY_PATTERNS.match(q):
        return True
    if len(set(q.lower())) < 3 and len(q) > 8:
        return True
    return False


def _is_jailbreak(question: str) -> bool:
    """
    Chống: DAN-style, Role-Play Override, Refusal Suppression,
           Few-shot Injection, Typo/Synonym Attack
    """
    return bool(_JAILBREAK_PATTERNS.search(question))


def _is_chitchat(question: str) -> bool:
    """Chỉ match nếu TOÀN BỘ câu là chitchat — chống Chitchat Bypass"""
    return bool(_CHITCHAT_PATTERNS.match(question.strip()))


# =============================================================================
# LAYER 1C — HISTORY AUDITOR
# Chống: Multi-turn / History Hijacking
# =============================================================================

def _audit_chat_history(chat_history: List[BaseMessage]) -> List[BaseMessage]:
    """
    Kiểm tra lịch sử hội thoại, loại bỏ các tin nhắn cố tình
    thiết lập lại danh tính hay quy tắc của bot.
    Giới hạn lịch sử để tránh context window poisoning.
    """
    if not chat_history:
        return []

    # Giới hạn tối đa 20 turns gần nhất
    MAX_HISTORY_TURNS = 20
    chat_history = chat_history[-MAX_HISTORY_TURNS:]

    cleaned = []
    for msg in chat_history:
        content = getattr(msg, 'content', '')
        if isinstance(content, str):
            # Kiểm tra xem tin nhắn có chứa jailbreak attempt không
            if _is_jailbreak(content):
                logger.warning(
                    f"[Security] Jailbreak attempt in history ({msg.__class__.__name__}): "
                    f"{content[:80]}..."
                )
                # Không loại bỏ hoàn toàn (sẽ gây lệch role sequence)
                # Thay thế nội dung nguy hiểm
                from langchain_core.messages import HumanMessage, AIMessage
                if isinstance(msg, HumanMessage):
                    cleaned.append(HumanMessage(content="[câu hỏi đã bị lọc]"))
                elif isinstance(msg, AIMessage):
                    cleaned.append(AIMessage(content="[phản hồi đã bị lọc]"))
                continue
        cleaned.append(msg)

    return cleaned


# =============================================================================
# LAYER 3 — OUTPUT VALIDATOR
# Chống: Data Exfiltration qua output
# =============================================================================

_SYSTEM_LEAK_PATTERNS = re.compile(
    r'(===BEGIN_DOC|===END_DOC|IDENTITY.*IMMUTABLE|SECURITY RULES|'
    r'BẢO MẬT.*BẮT BUỘC|system\s*prompt|instruct\s*me\s*to)',
    re.IGNORECASE,
)

def _validate_output(answer: str) -> str:
    """
    Kiểm tra output có vô tình leak system prompt hay raw context không.
    Nếu phát hiện, thay bằng câu trả lời an toàn.
    """
    if _SYSTEM_LEAK_PATTERNS.search(answer):
        logger.error("[Security] Output contains potential system prompt leak — blocked")
        return (
            "Xin lỗi, đã xảy ra lỗi xử lý. "
            "Vui lòng đặt câu hỏi của bạn theo cách khác."
        )
    return answer


# =============================================================================
# GENERATION SERVICE
# =============================================================================

class GenerationService:
    """
    Generation service với phòng thủ đa tầng:
    - Layer 1a: Input sanitizer (encoding, delimiter)
    - Layer 1b: Intent detector (jailbreak, invalid, chitchat)
    - Layer 1c: History auditor (multi-turn hijacking)
    - Layer 2:  Hardened system prompts (prompt-level defense)
    - Layer 3:  Output validator (exfiltration check)
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or OllamaLLMClient()
        logger.info("GenerationService (hardened) đã khởi tạo")

    def generate(
        self,
        retrieval_result: RetrievalResult,
        chat_history: List[BaseMessage] = None,
    ) -> str:

        # ── Layer 1a: Normalize & sanitize input ─────────────────────────────
        question = _decode_and_normalize(retrieval_result.query)

        # ── Layer 1c: Audit history ───────────────────────────────────────────
        chat_history = _audit_chat_history(chat_history or [])

        # ── Layer 1b: Intent detection (if/elif — chỉ một nhánh chạy) ────────
        if _is_invalid_query(question):
            prompt_type = PromptType.INVALID_QUERY
            prompt_vars = {"question": question}
            logger.info("[Generation] Intent: INVALID_QUERY")

        elif _is_jailbreak(question):
            # Jailbreak được xử lý bởi INVALID_QUERY prompt với câu hỏi trung tính
            prompt_type = PromptType.INVALID_QUERY
            prompt_vars = {"question": "Câu hỏi không hợp lệ"}
            logger.warning(f"[Security] Jailbreak attempt blocked: {question[:100]}")

        elif _is_chitchat(question):
            prompt_type = PromptType.CHITCHAT
            prompt_vars = {"question": question, "chat_history": chat_history}
            logger.info("[Generation] Intent: CHITCHAT")

        elif not retrieval_result.documents:
            prompt_type = PromptType.SIMPLE
            prompt_vars = {"question": question, "chat_history": chat_history}
            logger.info("[Generation] Intent: SIMPLE (không tìm thấy tài liệu)")

        else:
            # ── Layer 1a (cont): Sanitize document context ────────────────────
            context = _build_safe_context(retrieval_result.documents)
            prompt_type = PromptType.RAG
            prompt_vars = {
                "question": question,
                "context": context,
                "chat_history": chat_history,
            }
            logger.info(
                f"[Generation] Intent: RAG | Docs: {len(retrieval_result.documents)}"
            )

        # ── Layer 2: Build prompt & invoke LLM ───────────────────────────────
        prompt = PromptRegistry.get(prompt_type)
        messages = prompt.invoke(prompt_vars).messages
        logger.info(
            f"[Generation] Generating with history ({len(chat_history)} messages)"
        )
        raw_answer = self.llm_client.invoke(messages)

        # ── Layer 3: Validate output ──────────────────────────────────────────
        answer = _validate_output(raw_answer)

        return answer