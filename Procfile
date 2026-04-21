chatbot: python -m uvicorn app.api_bot:app --host 0.0.0.0 --port 7999 --reload
etl_api: python -m uvicorn app.api_etl:app --host 0.0.0.0 --port 8000 --reload
ocr_wrk: python pipeline/workers/ocr_worker.py
tomd_wrk: python pipeline/workers/to_md_worker.py
clean_wrk: python pipeline/workers/cleaning_worker.py
chunk_wrk: python pipeline/workers/chunking_worker.py
embed_wrk: python pipeline/workers/embedding_worker.py
