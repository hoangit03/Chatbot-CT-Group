import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.health import router as router_health
from app.api.v1.chat import router as router_chat

def create_app() -> FastAPI:
    """Factory pattern tạo FastAPI app cho Tầng Chatbot RAG."""
    app = FastAPI(
        title="ChatBot CT Group - Bot API",
        description="API chuyên dụng cho hệ thống RAG Chatbot",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router_health)
    app.include_router(router_chat)

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    print("==================================================================")
    print("[+] Khởi động Chatbot RAG Server")
    print("[+] API Docs: http://0.0.0.0:8000/docs")
    print("==================================================================")
    uvicorn.run("app.api_bot:app", host="0.0.0.0", port=8000, reload=True)
