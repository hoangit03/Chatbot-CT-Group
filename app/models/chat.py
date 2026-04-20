from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class Message(BaseModel):
    role: str          
    content: str


class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = None


class Source(BaseModel):
    file_name: str
    score: float


class ChatResponse(BaseModel):
    """Custom JSON Response theo yêu cầu"""
    success: bool = True
    query: str
    answer: str
    # sources: List[Source]
    # retrieved_count: int
    timestamp: str = datetime.now().isoformat()
    error: Optional[str] = None