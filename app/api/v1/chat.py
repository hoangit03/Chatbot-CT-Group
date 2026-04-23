from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List

from langchain_core.messages import HumanMessage, AIMessage

from app.models.chat import ChatRequest, ChatResponse, Message, Source
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["Chatbot"])

rag_service = RAGService()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    API Chatbot RAG - Hỗ trợ multi-turn conversation (Blocking - Chờ xong mới trả)
    """
    try:
        history = _build_history(request)

        # Gọi RAG Service
        result = rag_service.answer(
            query=request.query,
            chat_history=history
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

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


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