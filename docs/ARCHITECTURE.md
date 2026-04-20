# Kiến Trúc Hệ Thống RAG Phân Tán (Distributed RAG Architecture)

> **Mục tiêu:** Tối ưu hóa hóa luồng xử lý tài liệu Nhân sự (HR Data Pipeline) bằng cách tháo gỡ điểm nghẽn (bottleneck) tại Khâu Nạp OCR, phân bổ khối lượng tính toán dựa trên tiềm năng VRAM thực tế để tối đa hóa hiệu suất toàn hệ thống.

---

## 1. Sơ đồ Hoạt động Tổng thể

Luồng chảy của hệ thống vận hành theo quy tắc **"Điều Phối Cơ Động"** giữa 2 Cỗ máy. Máy 8GB đóng vai trò Bộ Não Điều Hành toàn diện (Lưu trữ + Hàng Đợi), còn Máy 4GB đóng vai trò Lõi Xử Lý Thị Giác Khổng Lồ độc lập.

Tóm tắt chu trình:
1. **User (Upload)** -> `API Endpoint (Node 8GB)` -> Gửi tác vụ vào `RabbitMQ`.
2. Nếu File là `PDF` hoặc `PPTX` (đã convert sang PDF):
   - `paddle_engine.py` (Node 8GB) nén file dưới dạng Base64.
   - Giao thức mạng TCP/IP "Push" cục Base64 sang cho VLM Server tại cổng `8080` (Node 4GB).
   - VLM Server (`PaddleOCR-VL-1.5`) bung nén, phân tích hình học, xử lý trọn vẹn Bảng/Biểu theo góc nhìn Multimodal và gửi ngược file `.MD` (Markdown) siêu sạch về lại cho Node 8GB.
3. Nếu File là `DOCX/XLSX`:
   - Trích xuất thẳng trên CPU/GPU bản địa của Node 8GB bằng `to_md_worker` (Vì file thô dễ nuốt, không cần OCR).
4. Chuỗi Hậu Kỳ (Hậu VLM) trên Node 8GB:
   - File `.MD` mới ra lò sẽ rới vào chuỗi Worker: `Cleaning Worker` -> `Chunking Worker` -> `Embedding Worker`.
   - Các Worker này sử dụng CPU/GPU 8GB để nghiền nát Vector và Lưu trữ vào VectorDB ngay trong nội bộ Node 8GB.

---

## 2. Nhiệm vụ Tách Biệt Toàn Diện

### 🌌 MÁY 1 (NODE 8GB VRAM) - Trạm Trung Tâm (The Orchestrator)
**Vai trò:** Trái tim điều phối Logic, tiền xử lý, gánh hệ sinh thái Database và Vector Embedder.
**Nhiệm vụ Chạy:**
- **RabbitMQ (Cổng 5672):** Bưu điện truyền phát Task tập trung.
- **ChromaDB Vector Store (Cổng 8002):** Lưu trữ Data tĩnh.
- `app/api_etl:app` (Giao diện Frontend nhận tệp của người dùng).
- Các Worker Đuôi (Code Python gốc): `ocr_worker`, `worker_to_md`, `cleaning_worker`, `chunking_worker`, `embedding_worker`.
  - `worker_to_md`: Đảm nhận việc trích xuất siêu tốc các tệp văn bản thô (DOCX, EXCEL, PPTX).
  - `ocr_worker`: Nếu tệp là PDF, worker này sẽ đóng gói tệp thành Base64 và đẩy sang máy 4GB.
- Embedding Model (E5-Large VRAM/CPU) phục vụ khâu nghiền Text -> Vector.

### 🏭 MÁY 2 (NODE 4GB VRAM) - Trạm Vệ Tinh VLM (The OCR Satellite)
**Vai trò:** Được cởi trói 100% RAM/VRAM để gánh trọn một Service siêu nặng.
**Nhiệm vụ Chạy:**
- **VLM API Server - PaddleX (Cổng 8080):** 
   Được Custom bằng `FastAPI` trong `app/api_paddle.py`. Ngốn 3.9/4GB VRAM để chứa model Vision-Language Mắt thần `PaddleOCR-VL-1.5`.

---

## 3. Cẩm Nang Vận Hành (Runbook)

Để đánh thức hệ thống, bạn cần thao tác trên hai máy theo tứ tự sau:

### Bước 1: Khởi động Trạm OCR Vệ Tinh (Trên MÁY 2 - 4GB)
1. Giữ File mã nguồn trong máy 4GB.
2. Tại thư mục `Chatbot-CT-Group`, mở file `docker-compose.yml`, chỉ cần giữ lại Core của `paddleocr` (có thể xóa khối rabbitmq/chroma đi để cực kỳ nhẹ máy).
3. Khởi chạy Container:
   ```bash
   docker compose up -d paddleocr
   ```
4. Theo dõi Log: `docker logs -f paddleocr_server` -> Chờ đến khi thấy chữ `Uvicorn running on http://0.0.0.0:8080`.

---

### Bước 2: Khởi động Lõi Cơ Sở Dữ Liệu (Trên MÁY 1 - 8GB)
1. Đảm bảo file `docker-compose.yml` trên máy này có 2 khối `rabbitmq` và `chromadb`.
2. Khởi chạy Backend lõi:
   ```bash
   docker compose up -d rabbitmq chromadb
   ```

---

### Bước 3: Thiết lập Giao Thức Mạng (Trên MÁY 1 - 8GB)
1. Mở file `.env` của thư mục Hệ thống trên Máy 8GB.
2. Thiết lập trỏ đường đi của RabbitMQ và Chroma vào Localhost, và chĩa họng súng OCR về phía trạm vệ tinh:
   ```env
   # .env
   # Vì MQ và Chroma chạy ngay tại máy này
   WORKER_NODE_IP=localhost
   RABBITMQ_HOST=localhost
   CHROMA_HOST=localhost
   
   # Địa chỉ IP LAN trỏ sang MÁY 4GB
   PADDLE_OCR_ENDPOINT=http://10.6.10.78:8080/layout-parsing
   ```

---

### Bước 4: Đánh Thức Trái Tim Điều Phối (Trên MÁY 1 - 8GB)
Chỉ khi MQ mở cửa, Máy 1 mới được kích hoạt các Worker Pika:

Cách dễ nhất (chỉ dùng trên Windows): Click đúp trực tiếp vào file **`start_pipeline.bat`**. Nó sẽ tự động bung ra 6 cửa sổ CLI chạy cho bạn:
- Web API (Cổng 8000)
- `ocr_worker`
- `to_md_worker`
- `cleaning_worker`
- `chunking_worker`
- `embedding_worker`

**✅ Bằng Chứng Thép:**
Hệ thống sẽ chạy một luồng tròn trịa: Mọi công việc bóc tách tài liệu nặng được đẩy sang Node 4GB làm "Culi chẻ củi", Node 8GB ung dung nhận thành quả Markdown siêu sạch, lưu Vector vào Chromadb, nhịp nhàng không một vệt lợn cợn!
