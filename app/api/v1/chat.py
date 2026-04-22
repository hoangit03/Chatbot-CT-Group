from fastapi import APIRouter, HTTPException
from typing import List, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage
from fastapi.responses import StreamingResponse
from app.models.chat import ChatRequest, ChatResponse, Message, Source
from app.services.rag_service import RAGService
import asyncio
import os
import json
import logging

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1", tags=["Chatbot"])
_REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 60))
rag_service = RAGService()

def _build_history(chat_history):
    history = []
    if chat_history:
        for msg in chat_history:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                history.append(AIMessage(content=msg.content))
    return history

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    API Chatbot RAG - Hỗ trợ multi-turn conversation
    """
    try:
        history = _build_history(request.chat_history)
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
        raise HTTPException(status_code=504, detail=f"Timeout sau {_REQUEST_TIMEOUT}s")
    except Exception as e:
        logger.error(f"[Chat] Error: {e}", exc_info=True)  # log full traceback
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Server-Sent Events — token xuất hiện ngay khi LLM generate.
    Không giảm tổng thời gian nhưng UX tốt hơn nhiều so với chờ 20s trắng.
 
    Response format:
      data: {"type": "token", "content": "Xin"}
      data: {"type": "token", "content": " chào"}
      data: {"type": "done", "sources": [...], "retrieved_count": 3}
      data: [DONE]
    """
    async def generate() -> AsyncIterator[str]:
        try:
            history = _build_history(request.chat_history)
 
            # Retrieval đầy đủ trước (không stream được)
            retrieval_result = await asyncio.wait_for(
                rag_service.retrieval.aretrieve(query=request.query),
                timeout=30.0,
            )
 
            # Build prompt (sync, <1ms)
            messages, original_lang = rag_service.generation._build_prompt(
                retrieval_result, history
            )
 
            # Stream LLM token-by-token
            async for token in rag_service.generation.llm_client.astream(messages):
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
 
            # Gửi metadata sau khi stream xong
            sources = [
                {
                    "file_name": doc.metadata.get("file_name"),
                    "score": float(
                        doc.metadata.get("rerank_score")
                        or doc.metadata.get("similarity_score", 0)
                    ),
                }
                for doc in retrieval_result.documents
            ]
            yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'retrieved_count': len(sources)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
 
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout'})}\n\n"
        except Exception as e:
            logger.error(f"[Stream] Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
 
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tắt nginx buffer nếu dùng nginx
        },
    )