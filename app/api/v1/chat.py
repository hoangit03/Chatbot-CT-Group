from fastapi import APIRouter, HTTPException, Header, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage

from app.models.chat import ChatRequest, ChatResponse, Message, Source
from app.services.rag_service import RAGService
from app.services.history_service import save_chat_message_async
import os

router = APIRouter(prefix="/api/v1", tags=["Chatbot"])

rag_service = RAGService()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    x_user_id: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None),
    x_role_level: Optional[str] = Header("1"),
    x_session_id: Optional[str] = Header(None)
):
    """
    API Chatbot RAG - Hỗ trợ multi-turn conversation (Blocking - Chờ xong mới trả)
    """
    try:
        history = _build_history(request)

        # Xây dựng Qdrant Filter dựa trên x_role_level
        from qdrant_client.http import models
        role_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.min_role_level",
                    range=models.Range(lte=int(x_role_level))
                )
            ]
        )

        # Lưu user message vào lịch sử
        if x_session_id:
            background_tasks.add_task(save_chat_message_async, x_session_id, "user", request.query, x_user_id, x_tenant_id)

        # Gọi RAG Service
        result = await rag_service.aanswer(
            query=request.query,
            chat_history=history,
            metadata_filter=role_filter
        )

        # Tạo response custom
        response = ChatResponse(
            success=True,
            query=request.query,
            answer=result["answer"],
            # sources=[Source(**s) for s in result["sources"]],
            # retrieved_count=result["retrieved_count"],
        )

        # Lưu bot message vào lịch sử
        if x_session_id:
            background_tasks.add_task(save_chat_message_async, x_session_id, "assistant", result["answer"], x_user_id, x_tenant_id)

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    x_user_id: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None),
    x_role_level: Optional[str] = Header("1"),
    x_session_id: Optional[str] = Header(None)
):
    """
    API Chatbot RAG - Streaming (Gõ từng chữ trả về cho UI)
    Trả về text/event-stream cho Streamlit hoặc bất kỳ SSE client nào.
    """
    try:
        history = _build_history(request)
        
        from qdrant_client.http import models
        role_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.min_role_level",
                    range=models.Range(lte=int(x_role_level))
                )
            ]
        )

        # Lưu user message vào lịch sử
        if x_session_id:
            background_tasks.add_task(save_chat_message_async, x_session_id, "user", request.query, x_user_id, x_tenant_id)

        # Streaming function wrap để có thể hook callback
        async def stream_and_save():
            full_response = ""
            async for chunk in rag_service.astream_answer(
                query=request.query, 
                chat_history=history,
                metadata_filter=role_filter
            ):
                if not chunk.startswith("<!-- thinking -->") and chunk != "<!-- clear_thinking -->":
                    full_response += chunk
                yield chunk
                
            # Lưu lịch sử sau khi stream xong
            if x_session_id and full_response:
                # Không dùng background_tasks ở đây được vì nó gắn với FastAPI route 
                # (đã exit scope lúc stream kết thúc), nên call luôn.
                await save_chat_message_async(x_session_id, "assistant", full_response, x_user_id, x_tenant_id)

        return StreamingResponse(
            stream_and_save(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_history(request: ChatRequest):
    """Helper: Convert chat_history từ JSON sang LangChain messages.
    
    QUAN TRỌNG: Giới hạn tối đa 3 messages gần nhất.
    Lý do: num_ctx=8192 tokens. History quá dài sẽ đẩy tài liệu RAG
    ra ngoài context window → LLM ảo giác.
    """
    MAX_HISTORY_MESSAGES = 3  # Tối đa 3 messages gần nhất

    
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
    
    print(f"\n{'='*60}")
    print(f"  📜 [HISTORY] {len(history)}/{len(request.chat_history or [])} messages (giới hạn {MAX_HISTORY_MESSAGES})")
    print(f"  📝 [QUERY] \"{request.query}\"")
    print(f"{'='*60}")
    if history:
        for i, msg in enumerate(history):
            role = "👤 User" if isinstance(msg, HumanMessage) else "🤖 Bot"
            snippet = msg.content[:150].replace('\n', ' ')
            print(f"  [{i+1}] {role}: \"{snippet}{'...' if len(msg.content) > 150 else ''}\"")
        print(f"{'='*60}")
    return history