from abc import ABC, abstractmethod
from typing import List, Optional
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


