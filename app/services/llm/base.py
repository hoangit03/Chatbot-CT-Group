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

    def stream(self, messages: List[BaseMessage]):
        """Default stream implementation — override for real streaming."""
        yield self.invoke(messages)
