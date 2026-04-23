from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

class BaseLLMClient(ABC):
    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        pass

    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> str:
        pass

    @abstractmethod
    async def ainvoke(self, messages: List[BaseMessage]) -> str:
        """Bất đồng bộ — dùng trong serve path (FastAPI).
        Subclass phải await asyncio.to_thread() hoặc native async call.
        """
        pass

    @abstractmethod
    def stream(self, messages: List[BaseMessage]):
        """Sync generator — yield từng text chunk."""
 
    @abstractmethod
    async def astream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """Async generator — yield từng text chunk (native async)."""

