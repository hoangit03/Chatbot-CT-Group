from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.health import router as router_health
from app.api.v1.chat import router as router_chat

def create_app()-> FastAPI:
    """Factory pattern tạo FastAPI app chính."""
    app = FastAPI(
        title="ChatBot CT Group",
        description="API for ChatBot CT Group",
        version="1.0.0",
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
    uvicorn.run(app, host="0.0.0.0", port=8000)