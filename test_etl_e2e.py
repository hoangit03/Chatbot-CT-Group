import sys
import os
import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(REPO_ROOT)

print("=== BẮT ĐẦU TEST E2E PIPELINE TRẢ KẾT QUẢ VỀ 8GB ===")
file_path = "d:/CTGroup/CT_Knowledge/report.pdf"
from pipeline.engines.paddle_engine import run_ocr_pipeline

print("1. Chạy OCR Engine tới máy VLM 4GB...")
res = run_ocr_pipeline("report.pdf")
print("Kết quả chạy:", res)

# Read the output markdown
MARKDOWN_OUTPUT_FILE = os.path.join(REPO_ROOT, "shared_data", "data_output", "result", "report_paddle_only.md")
if os.path.exists(MARKDOWN_OUTPUT_FILE):
    with open(MARKDOWN_OUTPUT_FILE, "r", encoding="utf-8") as f:
        md = f.read()
    print(f"File sinh ra: {MARKDOWN_OUTPUT_FILE}")
    print(f"Độ dài Markdown thu được: {len(md)} ký tự")
    print("--- TRÍCH ĐOẠN 200 KÝ TỰ ĐẦU ---")
    print(md[:200])
else:
    print("KHÔNG THẤY FILE KẾT QUẢ! ERROR!")
