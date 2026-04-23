@echo off
echo ===== BOOTING CT-GROUP RAG PIPELINE (8GB NODE) =====
echo.

set PYTHONPATH=%cd%

echo 1. Starting FastAPI ETL Orchestrator...
start cmd /k ".\.venv\Scripts\activate && uvicorn app.api_etl:app --host 0.0.0.0 --port 8000 --reload"

echo 3. Starting TO_MD Worker (Native CPU Parsing)...
start cmd /k ".\.venv\Scripts\activate && python pipeline\workers\to_md_worker.py"

echo 4. Starting CLEANING Worker...
start cmd /k ".\.venv\Scripts\activate && python pipeline\workers\cleaning_worker.py"

echo 5. Starting CHUNKING Worker...
start cmd /k ".\.venv\Scripts\activate && python pipeline\workers\chunking_worker.py"

echo 6. Starting EMBEDDING Worker...
start cmd /k ".\.venv\Scripts\activate && python pipeline\workers\embedding_worker.py"

echo All background workers initiated! (Including OCR Worker)
exit