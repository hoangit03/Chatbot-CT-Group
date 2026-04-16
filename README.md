# Chatbot CT-Group (RAG-based)


---

## 🛠️ Công nghệ
- Backend: FastAPI  
- LLM: Ollama  
- Framework: LangChain  
- Vector DB: Chroma 
- Embedding: Sentence Transformers  

---

## ⚙️ Cài đặt: nhớ upgrade pip

```bash
git clone <Chatbot-CT-Group>
cd chatbot-ct-group

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements-dev.txt

```
## Quy Trình Khởi Chạy Sau Tích Hợp

```powershell
# Bước 1: Khởi động Docker (PaddleOCR + RabbitMQ)
docker-compose up -d

# Bước 2: Chạy FastAPI Server (duy nhất 1 cổng)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Bước 3: Khởi động Workers (mỗi worker 1 terminal)
python -m pipeline.workers.ocr_worker
python -m pipeline.workers.to_md_worker
python -m pipeline.workers.cleaning_worker
python -m pipeline.workers.chunking_worker
python -m pipeline.workers.embedding_worker
```

**Endpoints có sẵn:**
| Method | Endpoint | Chức năng | Nguồn |
|---|---|---|---|
| POST | `/api/v1/chat` | Chat RAG | Chatbot |
| GET | `/api/v1/health` | Health check | Chatbot |
| POST | `/api/v1/extract` | Upload & auto-route ETL | XuLy |
| GET | `/api/v1/vectordb/all` | Debug VectorDB | XuLy |
| DELETE | `/api/v1/vectordb/all` | Clear VectorDB | XuLy |

---

## Lưu Đồ Luồng Dữ Liệu Sau Tích Hợp

```mermaid
flowchart TD
    subgraph FRONTEND["🖥️ Frontend / Client"]
        User["Người dùng HR"]
    end

    subgraph API_LAYER["🌐 FastAPI Server (Port 8000)"]
        ChatAPI["POST /api/v1/chat"]
        ExtractAPI["POST /api/v1/extract"]
        VectorAPI["GET /api/v1/vectordb/all"]
    end

    subgraph RAG_SERVICES["🧠 RAG Services (app/services/)"]
        RAG["RAGService"]
        Retrieval["RetrievalService + Reranker"]
        Generation["GenerationService (Ollama)"]
        Embedder["Embedder (e5-large)"]
    end

    subgraph PIPELINE["⚙️ ETL Pipeline (pipeline/)"]
        MQ{"RabbitMQ"}
        W1["OCR Worker"]
        W2["ToMD Worker"]
        WC["Cleaning Worker"]
        W3["Chunking Worker"]
        W4["Embedding Worker"]
        E1["paddle_engine"]
        E2["to_md_engine"]
        EC["cleaning_engine"]
        E3["chunking_engine"]
        E4["embedding_engine"]
    end

    VDB[("ChromaDB\nvectorstore/chroma_db/")]

    User -->|"Hỏi đáp"| ChatAPI
    User -->|"Upload tài liệu"| ExtractAPI
    User -->|"Debug"| VectorAPI

    ChatAPI --> RAG
    RAG --> Retrieval
    Retrieval --> Embedder
    Retrieval --> VDB
    Retrieval --> Generation

    ExtractAPI -->|"Publish task"| MQ
    MQ --> W1 & W2
    W1 --> E1
    W2 --> E2
    E1 & E2 -->|"MD output"| WC
    WC --> EC
    EC -->|"Clean MD"| W3
    W3 --> E3
    E3 -->|"Chunks JSON"| W4
    W4 --> E4
    E4 -->|"Embed & Store"| VDB

    VectorAPI --> VDB
```

---


