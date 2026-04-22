import logging
import os
from typing import List, Optional, AsyncIterator
from dotenv import load_dotenv

from app.core.exception.llm_exception import LLMException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from app.services.llm.base import BaseLLMClient
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_LLM = int(os.getenv("LLM_MAX_CONCURRENT", 2))
_llm_semaphore: Optional[asyncio.Semaphore] = None

def _get_semaphore() -> asyncio.Semaphore:
    """Lazy-init trong running event loop — an toàn với uvicorn reload."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LLM)
    return _llm_semaphore

class OllamaLLMClient(BaseLLMClient):
    """Ollama client với auto-pull model + logging + exception rõ ràng"""

    def __init__(self):
        self.model_name = os.getenv("MODEL_LLM", "qwen3:4b")
        self.temperature = float(os.getenv("TEMPERATURE", 0.0))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", 8192))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", 512))
        self.num_thread = int(os.getenv("OLLAMA_NUM_THREAD", 0))
        self._llm: Optional[BaseChatModel] = None
        logger.info(f"[LLM] Khởi tạo Ollama client với model: {self.model_name}")

    def _ensure_model_exists(self):
        """Tự động pull model nếu chưa có"""
        import subprocess
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',      # ← Fix Unicode
                timeout=15
            )
            if self.model_name not in result.stdout:
                logger.info(f"[LLM] Model {self.model_name} chưa tồn tại. Đang pull...")
                print(f"Đang pull model {self.model_name} từ Ollama (có thể mất vài phút)...")
                
                pull_result = subprocess.run(
                    ["ollama", "pull", self.model_name],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',      # ← Fix UnicodeDecodeError
                    timeout=1800
                )
                if pull_result.returncode != 0:
                    raise LLMException(
                        message=f"Không thể pull model {self.model_name}",
                        model=self.model_name,
                        original_error=Exception(pull_result.stderr)
                    )
                logger.info(f"[LLM] Pull model {self.model_name} thành công")
        except FileNotFoundError:
            raise LLMException("Không tìm thấy lệnh 'ollama'. Vui lòng cài Ollama trước.")
        except Exception as e:
            raise LLMException("Lỗi khi kiểm tra/pull model Ollama", self.model_name, e)

    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._ensure_model_exists()
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                num_ctx=self.num_ctx,      # context lớn để hỗ trợ history dài
                num_predict=self.num_predict,
                num_thread=self.num_thread if self.num_thread > 0 else None,
            )
        return self._llm


    def invoke(self, messages: List[BaseMessage]) -> str:
        try:
            llm = self.get_llm()
            logger.debug(f"[LLM] Gửi prompt với {len(messages)} messages")
            response = llm.invoke(messages)
            logger.info(f"[LLM] Nhận được response ({len(response.content)} ký tự)")
            return response.content
        except LLMException:
            raise
        except Exception as e:
            raise LLMException("Lỗi khi gọi Ollama", self.model_name, e)
        
    async def ainvoke(self, messages: List[BaseMessage]) -> str:
         sem = _get_semaphore()
         async with sem:
            logger.debug(
                f"[LLM] async invoke | {len(messages)} messages | "
                f"semaphore={sem._value}/{_MAX_CONCURRENT_LLM}"  # type: ignore[attr-defined]
            )
            return await asyncio.to_thread(self.invoke, messages)
         
    async def astream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """Native async streaming — không cần thread."""
        sem = _get_semaphore()
        async with sem:
            llm = self.get_llm()
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content