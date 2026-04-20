import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.health import router as router_health
from app.api.v1.chat import router as router_chat
from app.api.v1.extract import router as router_extract
from app.api.v1.vectordb import router as router_vectordb

def create_app()-> FastAPI:
    """Factory pattern tạo FastAPI app chính."""
    app = FastAPI(
        title="ChatBot CT Group",
        description="API for ChatBot CT Group + ETL Data Pipeline",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # === Chatbot Routers ===
    app.include_router(router_health)
    app.include_router(router_chat)

    # === ETL Pipeline Routers (từ XuLy_Data_Rag_CTG) ===
    app.include_router(router_extract)
    app.include_router(router_vectordb)

    # === Static files cho pipeline output ===
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))
    OUTPUT_DIR = os.path.join(SHARED_DIR, "data_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    print("==================================================================")
    print("[+] Khởi động ChatBot CT Group + ETL Pipeline")
    print("[+] API Docs: http://0.0.0.0:8000/docs")
    print("==================================================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)