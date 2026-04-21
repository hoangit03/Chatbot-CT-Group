@echo off
echo ===== BOOTING CT-GROUP RAG PIPELINE (ALL-IN-ONE TERMINAL) =====
echo.

set PYTHONPATH=%cd%

echo (Thiết lập cấu hình Encoding chống lỗi hiển thị Tiếng Việt)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo (Đảm bảo thư viện gom màn hình honcho đã cài đặt)
call .\venv\Scripts\activate 2>nul || call .\.venv\Scripts\activate 2>nul
pip install honcho -q

echo.
echo ==============================================================
echo [!] Dang khoi dong toan bo Pipeline trong 1 man hinh duy nhat...
echo [!] Nhan To-hop phim [Ctrl + C] de Tat TOAN BO he thong nhe.
echo ==============================================================

honcho start -c clean_wrk=2,chunk_wrk=2,embed_wrk=2
