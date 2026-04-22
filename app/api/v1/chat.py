from fastapi import APIRouter, HTTPException
from typing import List
import asyncio

from langchain_core.messages import HumanMessage, AIMessage

from app.models.chat import ChatRequest, ChatResponse, Message, Source
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["Chatbot"])

rag_service = RAGService()

_REQUEST_TIMEOUT = float(60)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    API Chatbot RAG - Hỗ trợ multi-turn conversation
    """
    try:
        history = []
        if request.chat_history:
            for msg in request.chat_history:
                if msg.role == "user":
                    history.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    history.append(AIMessage(content=msg.content))

        # Gọi RAG Service
        result = await asyncio.wait_for(
            rag_service.aanswer(query=request.query, chat_history=history),
            timeout=_REQUEST_TIMEOUT,
        )

        # Tạo response custom
        response = ChatResponse(
            success=True,
            query=request.query,
            answer=result["answer"],
            # sources=[Source(**s) for s in result["sources"]],
            # retrieved_count=result["retrieved_count"],
        )

        return response

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Request timeout sau {_REQUEST_TIMEOUT}s. Vui lòng thử lại.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )