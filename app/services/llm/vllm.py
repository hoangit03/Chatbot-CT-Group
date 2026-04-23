import logging
import os
from typing import List, Optional, AsyncIterator
from dotenv import load_dotenv
import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.services.llm.base import BaseLLMClient
from app.core.exception.llm_exception import LLMException

load_dotenv()
logger = logging.getLogger(__name__)

_MAX_CONCURRENT_LLM = int(os.getenv("LLM_MAX_CONCURRENT", 8))
_llm_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LLM)
    return _llm_semaphore


class VLLMClient(BaseLLMClient):
    """
    vLLM Client dùng OpenAI-compatible endpoint.

    vLLM tự xử lý batching + continuous batching bên trong — tốt hơn nhiều so với
    Ollama khi có 500+ concurrent requests. Client chỉ cần gửi HTTP requests bình thường.

    Ưu điểm so với OllamaLLMClient:
    - Hỗ trợ PagedAttention → throughput cao hơn 10-20x
    - Built-in request queue — không cần semaphore nhiều như Ollama
    - Native async với aiohttp — không cần asyncio.to_thread()
    """

    def __init__(self):
        self.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = os.getenv("VLLM_API_KEY", "token-abc123")
        self.model_name = os.getenv("MODEL_LLM", "Qwen/Qwen3-4B")
        self.temperature = float(os.getenv("TEMPERATURE", 0.0))
        self.max_tokens = int(os.getenv("VLLM_MAX_TOKENS", 512))
        self._llm: Optional[BaseChatModel] = None
        logger.info(f"[vLLM] Khởi tạo client | model={self.model_name} | url={self.base_url}")

    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise LLMException(
                    "langchain-openai chưa được cài. Chạy: pip install langchain-openai"
                )
            self._llm = ChatOpenAI(
                model=self.model_name,
                openai_api_base=self.base_url,
                openai_api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # Tắt retry mặc định — xử lý retry ở tầng service nếu cần
                max_retries=1,
            )
            logger.info(f"[vLLM] LLM ready: {self.model_name} @ {self.base_url}")
        return self._llm

    def invoke(self, messages: List[BaseMessage]) -> str:
        import time
        try:
            llm = self.get_llm()
            t0 = time.time()
            response = llm.invoke(messages)
            elapsed = time.time() - t0
            char_count = len(response.content)
            logger.info(f"[vLLM] invoke | {char_count} chars | {elapsed:.2f}s")
            return response.content
        except LLMException:
            raise
        except Exception as e:
            raise LLMException("Lỗi khi gọi vLLM", self.model_name, e)

    async def ainvoke(self, messages: List[BaseMessage]) -> str:
        """
        Native async — vLLM/OpenAI client hỗ trợ async natively, không cần to_thread().
        Semaphore vẫn giữ để chống burst quá lớn gây OOM trên GPU.
        """
        sem = _get_semaphore()
        async with sem:
            try:
                llm = self.get_llm()
                response = await llm.ainvoke(messages)
                return response.content
            except LLMException:
                raise
            except Exception as e:
                raise LLMException("Lỗi async khi gọi vLLM", self.model_name, e)

    def stream(self, messages: List[BaseMessage]):
        try:
            llm = self.get_llm()
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise LLMException("Lỗi stream vLLM", self.model_name, e)

    async def astream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        sem = _get_semaphore()
        async with sem:
            llm = self.get_llm()
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content