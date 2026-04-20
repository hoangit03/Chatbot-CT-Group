# Kiến Trúc Hệ Thống RAG Phân Tán (Distributed RAG Architecture)

> **Mục tiêu:** Tối ưu hóa hóa luồng xử lý tài liệu Nhân sự (HR Data Pipeline) bằng cách tháo gỡ điểm nghẽn (bottleneck) tại Khâu Nạp OCR, phân bổ khối lượng tính toán dựa trên tiềm năng VRAM thực tế để tối đa hóa hiệu suất toàn hệ thống.

---

## 1. Sơ đồ Hoạt động Tổng thể

Luồng chảy của hệ thống vận hành theo quy tắc **"Điều Phối Cơ Động"** giữa 2 Cỗ máy. Máy 8GB đóng vai trò Bộ Não Điều Hành, còn Máy 4GB đóng vai trò Lõi Xử Lý Cơ Sở Dữ Liệu và Thị Giác Khổng Lồ.

Tóm tắt chu trình:
1. **User (Upload)** -> `API Endpoint (Node 8GB)` -> Gửi tác vụ vào `RabbitMQ`.
2. Nếu File là `PDF` hoặc `PPTX` (đã convert sang PDF):
   - `paddle_engine.py` (Node 8GB) nén file dưới dạng Base64.
   - Giao thức mạng TCP/IP "Push" cục Base64 sang cho VLM Server tại cổng `8080` (Node 4GB).
   - VLM Server (`PaddleOCR-VL-1.5`) bung nén, phân tích hình học, xử lý trọn vẹn Bảng/Biểu theo góc nhìn Multimodal và gửi ngược file `.MD` (Markdown) siêu sạch về lại cho Node 8GB.
3. Nếu File là `DOCX/XLSX`:
   - Trích xuất thẳng trên CPU/GPU bản địa của Node 8GB (Vì file thô dễ nuốt, không cần OCR).
4. Chuỗi Hậu Kỳ (Hậu VLM) trên Node 8GB:
   - File `.MD` mới ra lò sẽ rới vào chuỗi Worker: `Cleaning Worker` -> `Chunking Worker` -> `Embedding Worker`.
   - Các Worker này sử dụng CPU/GPU 8GB để nghiền nát Vector và Ping qua cổng `8002` Lưu trữ thẻ nhớ vào VectorDB vĩnh trú trên Node 4GB.

---

## 2. Nhiệm vụ Tách Biệt Toàn Diện

### 🌌 MÁY 1 (NODE 8GB VRAM) - The Orchestrator
**Vai trò:** Trái tim điều phối Logic, tiền xử lý và đẩy Vector cực tinh vi.
**Nhiệm vụ Chạy:**
- `app/api_etl:app` (Giao diện Frontend nhận tệp của người dùng).
- Các Worker Đuôi (Code Python gốc): `parser_worker`, `cleaning_worker`, `chunking_worker`, `embedding_worker`.
- Embedding Model (E5-Large VRAM/CPU) phục vụ khâu nghiền Text -> Vector.

### 🏭 MÁY 2 (NODE 4GB VRAM) - The Infrastructure
**Vai trò:** Cái nôi Lõi Phần cứng ngầm, gánh trọn các Service Thặng Dư (Heavy Docker).
**Nhiệm vụ Chạy:**
- **RabbitMQ (Cổng 5672):** Bưu điện truyền phát Task cho 2 máy.
- **ChromaDB Vector Store (Cổng 8002):** Lưu trữ Data tĩnh.
- **VLM API Server - PaddleX (Cổng 8080):** 
   Được Custom bằng `FastAPI` trong `app/api_paddle.py`. Nó chứa model Vision-Language Mắt thần `PaddleOCR-VL-1.5`.

---

## 3. Cẩm Nang Vận Hành (Runbook)

Để đánh thức hệ thống, bạn cần thao tác trên hai máy theo tứ tự sau:

### Bước 1: Khởi động Đáy Tầng (Trên MÁY 2 - 4GB)
> (Bắt buộc phải lên máy này bật trước để RabbitMQ và OCR Server thức giấc)

1. Giữ File mã nguồn trong máy 4GB như hiện tại.
2. Tại thư mục `Chatbot-CT-Group`, khởi run các Container Lõi bằng Docker Compose:
   ```bash
   docker compose up -d rabbitmq chromadb paddleocr
   ```
3. Theo dõi Log của thằng khổng lồ OCR: 
   ```bash
   docker logs -f paddleocr_server
   ```
   **Dấu hiệu thành công:** Chờ đến khi thấy chữ `Uvicorn running on http://0.0.0.0:8080` (Model Weights có thể tải rất lâu trong lần đầu).

---

### Bước 2: Thiết lập Giao Thức Mạng (Trên MÁY 1 - 8GB)
1. Giữ Code cập nhật mới nhất ở nhánh `dev/distributed-architecture`.
2. Mở file `.env` của thư mục Hệ thống trên Máy 8GB.
3. Chỉnh cái thông số Xương Sống (`WORKER_NODE_IP`) trỏ vào địa chỉ IP LAN của Máy 4GB (VD: Máy 4GB mạng công ty IP là `192.168.1.150` thì thảy nó vào).
   ```env
   # .env
   WORKER_NODE_IP=192.168.1.150
   PADDLE_OCR_ENDPOINT=http://${WORKER_NODE_IP}:8080/layout-parsing
   CHROMA_HOST=${WORKER_NODE_IP}
   RABBITMQ_HOST=${WORKER_NODE_IP}
   ```
   *(Hệ thống của tôi đã tự động config để Python đọc và thông nối các Port `8080`, `8002`, `5672` theo sát cái Base IP này).*

---

### Bước 3: Đánh Thức Trái Tim Điều Phối (Trên MÁY 1 - 8GB)
Chỉ khi Bước 1 chạy xong, Bưu điện MQ mở cửa, Máy 1 mới được kích hoạt:

Bật các terminal tại thư mục Code và chạy Backend ETL + Rabbit Workers:
```bash
# Terminal 1: Bật API Đón File từ Client
uvicorn app.api_etl:app --port 8001 

# Terminal 2, 3, 4: Bật thợ phu
python -m pipeline.workers.ocr_worker
python -m pipeline.workers.cleaning_worker
python -m pipeline.workers.chunking_worker
python -m pipeline.workers.embedding_worker
```

### ✅ Bằng chứng Thép
Hệ thống sẽ chạy một luồng tròn trịa:
- Máy `8GB` ném cục PDF vào File. Đọc Base64, PING thẳng qua IP `192.168...:8080`.
- Máy `4GB` khè ra Lửa VRAM để đọc Ảnh bằng VLM AI, ghép Markdown siêu cấp gửi trả.
- Máy `8GB` nhận lại `.md`, đập `#` và gom vào Vector Embedder.
- Xong gọi PING sang `192.168...:8002` cất tủ bên `4GB`. 

Sạch sẽ, Phân công rõ ràng, Tận dụng 12GB VRAM hợp lực không lãng phí 1MB nào!
