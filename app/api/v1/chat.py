from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import asyncio
import json
import time

from langchain_core.messages import HumanMessage, AIMessage
from fastapi.responses import StreamingResponse
from app.models.chat import ChatRequest, ChatResponse, Message, Source
from app.services.rag_service import RAGService
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import logging

router = APIRouter(prefix="/api/v1", tags=["Chatbot"])
logger = logging.getLogger(__name__)
rag_service = RAGService()

_REQUEST_TIMEOUT = float(60)

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
    API Chatbot RAG - Hỗ trợ multi-turn conversation (Blocking - Chờ xong mới trả)
    """
    try:
        time_start = time.time()
        history = _build_history(request)
        # Gọi RAG Service
        
        result = await asyncio.wait_for(
            rag_service.aanswer(query=request.query, chat_history=history),
            timeout=_REQUEST_TIMEOUT,
        )
        time_end = time.time()
        # Tạo response custom
        response = ChatResponse(
            success=True,
            query=request.query,
            answer=result["answer"],
            total_time= round(time_end - time_start, 6)
            # sources=[Source(**s) for s in result["sources"]],
            # retrieved_count=result["retrieved_count"],
        )

        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Timeout sau {_REQUEST_TIMEOUT}s")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# @router.post("/chat/stream")
# async def chat_stream(request: ChatRequest):
#     """
#     Server-Sent Events — token xuất hiện ngay khi LLM generate.
#     Không giảm tổng thời gian nhưng UX tốt hơn nhiều so với chờ 20s trắng.
 
#     Response format:
#       data: {"type": "token", "content": "Xin"}
#       data: {"type": "token", "content": " chào"}
#       data: {"type": "done", "sources": [...], "retrieved_count": 3}
#       data: [DONE]
#     """
#     async def generate() -> AsyncIterator[str]:
#         try:
#             history = _build_history(request)
 
#             # Retrieval đầy đủ trước (không stream được)
#             retrieval_result = await asyncio.wait_for(
#                 rag_service.retrieval.aretrieve(query=request.query),
#                 timeout=_REQUEST_TIMEOUT,
#             )
 
#             # Build prompt (sync, <1ms)
#             messages, original_lang = rag_service.generation._build_prompt(
#                 retrieval_result, history
#             )
 
#             # Stream LLM token-by-token
#             async for token in rag_service.generation.llm_client.astream(messages):
#                 yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
 
#             # Gửi metadata sau khi stream xong
#             sources = [
#                 {
#                     "file_name": doc.metadata.get("file_name"),
#                     "score": float(
#                         doc.metadata.get("rerank_score")
#                         or doc.metadata.get("similarity_score", 0)
#                     ),
#                 }
#                 for doc in retrieval_result.documents
#             ]
#             yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'retrieved_count': len(sources)}, ensure_ascii=False)}\n\n"
#             yield "data: [DONE]\n\n"
 
#         except asyncio.TimeoutError:
#             yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout'})}\n\n"
#         except Exception as e:
#             logger.error(f"[Stream] Error: {e}", exc_info=True)
#             yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
 
#     return StreamingResponse(
#         generate(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "X-Accel-Buffering": "no",  # tắt nginx buffer nếu dùng nginx
#         },
#     )


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    API Chatbot RAG - Streaming (Gõ từng chữ trả về cho UI)
    Trả về text/event-stream cho Streamlit hoặc bất kỳ SSE client nào.
    """
    try:
        history = _build_history(request)
        return StreamingResponse(
            rag_service.stream_answer(query=request.query, chat_history=history),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_history(request: ChatRequest):
    """Helper: Convert chat_history từ JSON sang LangChain messages.
    
    QUAN TRỌNG: Giới hạn tối đa 6 messages gần nhất (3 cặp Q&A).
    Lý do: num_ctx=4096 tokens. Nếu nhồi 20+ messages lịch sử,
    tài liệu RAG sẽ bị đẩy ra ngoài context window → LLM ảo giác.
    """
    MAX_HISTORY_MESSAGES = 6  # 3 cặp Q&A = vừa đủ ngữ cảnh
    
    history = []
    if request.chat_history:
        messages_to_process = list(request.chat_history)
        # Cắt bỏ tin nhắn user cuối cùng nếu trùng với query hiện tại
        if (messages_to_process 
            and messages_to_process[-1].role == "user" 
            and messages_to_process[-1].content.strip() == request.query.strip()):
            messages_to_process = messages_to_process[:-1]
        
        # Chỉ lấy N messages gần nhất
        messages_to_process = messages_to_process[-MAX_HISTORY_MESSAGES:]
        
        for msg in messages_to_process:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                # Cắt ngắn câu trả lời cũ để tiết kiệm token
                content = msg.content[:300] if len(msg.content) > 300 else msg.content
                history.append(AIMessage(content=content))
    
    print(f"  📜 [History] {len(history)}/{len(request.chat_history or [])} messages (giới hạn {MAX_HISTORY_MESSAGES})")
    return history
