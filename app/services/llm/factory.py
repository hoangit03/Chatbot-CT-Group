import os
import logging
from dotenv import load_dotenv

from app.services.llm.base import BaseLLMClient

load_dotenv()
logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory trả về LLM client theo LLM_PROVIDER env var.
    Lazy import — chỉ load thư viện khi thực sự dùng.
    """

    _registry: dict[str, type] = {}
    _instance: BaseLLMClient | None = None  # Singleton per process

    @classmethod
    def _ensure_defaults_registered(cls) -> None:
        """Đăng ký built-in providers nếu chưa có (tránh circular import khi module load)."""
        if "ollama" not in cls._registry:
            from app.services.llm.ollama import OllamaLLMClient
            cls._registry["ollama"] = OllamaLLMClient

        if "vllm" not in cls._registry:
            from app.services.llm.vllm import VLLMClient
            cls._registry["vllm"] = VLLMClient

    @classmethod
    def register(cls, name: str, client_class: type) -> None:
        """
        Đăng ký provider custom từ bên ngoài.

        Ví dụ:
            from app.services.llm.factory import LLMFactory
            LLMFactory.register("anthropic", AnthropicLLMClient)
        """
        if not issubclass(client_class, BaseLLMClient):
            raise TypeError(f"{client_class} phải kế thừa BaseLLMClient")
        cls._registry[name.lower()] = client_class
        logger.info(f"[LLMFactory] Đã đăng ký provider: '{name}'")

    @classmethod
    def get_client(cls, *, force_new: bool = False) -> BaseLLMClient:
        """
        Trả về singleton LLM client.

        Args:
            force_new: Tạo instance mới (dùng trong tests hoặc hot-reload config).
        """
        if cls._instance is not None and not force_new:
            return cls._instance

        cls._ensure_defaults_registered()

        provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        if provider not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"LLM_PROVIDER='{provider}' không hợp lệ. "
                f"Các giá trị hợp lệ: {available}"
            )

        client_class = cls._registry[provider]
        cls._instance = client_class()
        logger.info(f"[LLMFactory] Khởi tạo LLM client: provider='{provider}' class={client_class.__name__}")
        return cls._instance