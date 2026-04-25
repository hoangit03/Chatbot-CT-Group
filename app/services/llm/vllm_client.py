"""
vLLM LLM Client — OpenAI-compatible API client for vLLM server.
Thay thế OllamaLLMClient khi deploy trên server có vLLM.
"""
import logging
import os
import time
from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from app.services.llm.base import BaseLLMClient
from app.core.exception.llm_exception import LLMException

load_dotenv()

logger = logging.getLogger(__name__)


class VLLMClient(BaseLLMClient):
    """vLLM client — gọi qua OpenAI-compatible API (nhanh hơn Ollama)"""

    def __init__(self):
        self.model_name = os.getenv("MODEL_LLM", "Qwen/Qwen2.5-3B-Instruct")
        self.base_url = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
        self.temperature = float(os.getenv("TEMPERATURE", 0.1))
        self._llm: Optional[BaseChatModel] = None
        logger.info(f"[LLM] Khởi tạo vLLM client: model={self.model_name}, url={self.base_url}")

    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                base_url=self.base_url,
                api_key="no-key",
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=1024,
            )
            logger.info(f"[LLM] vLLM ChatOpenAI client ready")
        return self._llm

    def invoke(self, messages: List[BaseMessage]) -> str:
        try:
            llm = self.get_llm()
            logger.debug(f"[LLM] Gửi prompt với {len(messages)} messages")

            t0 = time.time()
            response = llm.invoke(messages)
            t_llm = time.time() - t0

            char_count = len(response.content)
            token_est = char_count // 4
            speed = token_est / t_llm if t_llm > 0 else 0

            print(f"  ⏱️  [vLLM] Sinh {char_count} ký tự (~{token_est} tokens) trong {t_llm:.2f}s ({speed:.1f} tok/s)")
            logger.info(f"[LLM] Response ({char_count} ký tự) trong {t_llm:.2f}s")
            return response.content
        except Exception as e:
            raise LLMException("Lỗi khi gọi vLLM", self.model_name, e)

    def stream(self, messages: List[BaseMessage]):
        """Generator: Yield từng chunk text từ vLLM (streaming)"""
        try:
            llm = self.get_llm()
            logger.debug(f"[LLM Stream] Gửi prompt với {len(messages)} messages")
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise LLMException("Lỗi khi stream vLLM", self.model_name, e)
