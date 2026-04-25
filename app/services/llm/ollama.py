import logging
import os
from typing import List, Optional
from dotenv import load_dotenv

from app.core.exception.llm_exception import LLMException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from app.services.llm.base import BaseLLMClient

load_dotenv()

logger = logging.getLogger(__name__)

class OllamaLLMClient(BaseLLMClient):
    """Ollama client với auto-pull model + logging + exception rõ ràng"""

    def __init__(self):
        self.model_name = os.getenv("MODEL_LLM", "qwen3:4b")
        self.temperature = float(os.getenv("TEMPERATURE", 0.0))
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
                num_ctx=8192,       # System(3K) + History(1.5K) + Docs(2K) + Output(1K) = ~7.5K
                num_predict=1024,   # Giới hạn output tối đa 1024 tokens (~800 từ TV)
            )
        return self._llm

    def invoke(self, messages: List[BaseMessage]) -> str:
        import time
        try:
            llm = self.get_llm()
            logger.debug(f"[LLM] Gửi prompt với {len(messages)} messages")
            
            t0 = time.time()
            response = llm.invoke(messages)
            t_llm = time.time() - t0
            
            char_count = len(response.content)
            token_est = char_count // 4  # Ước tính token cho tiếng Việt
            speed = token_est / t_llm if t_llm > 0 else 0
            
            print(f"  ⏱️  [LLM] Sinh {char_count} ký tự (~{token_est} tokens) trong {t_llm:.2f}s ({speed:.1f} tok/s)")
            logger.info(f"[LLM] Nhận được response ({char_count} ký tự)")
            return response.content
        except Exception as e:
            raise LLMException("Lỗi khi gọi Ollama", self.model_name, e)

    def stream(self, messages: List[BaseMessage]):
        """Generator: Yield từng chunk text từ LLM (dùng cho Streaming)"""
        try:
            llm = self.get_llm()
            logger.debug(f"[LLM Stream] Gửi prompt với {len(messages)} messages")
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise LLMException("Lỗi khi stream Ollama", self.model_name, e)

    async def ainvoke(self, messages: List[BaseMessage]) -> str:
        import time
        try:
            llm = self.get_llm()
            t0 = time.time()
            response = await llm.ainvoke(messages)
            t_llm = time.time() - t0
            char_count = len(response.content)
            logger.info(f"[LLM Async] Nhận được response ({char_count} ký tự) trong {t_llm:.2f}s")
            return response.content
        except Exception as e:
            raise LLMException("Lỗi khi gọi Ollama async", self.model_name, e)

    async def astream(self, messages: List[BaseMessage]):
        try:
            llm = self.get_llm()
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise LLMException("Lỗi stream Ollama async", self.model_name, e)