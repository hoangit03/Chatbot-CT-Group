# Nhật ký Tối ưu hóa Hệ thống RAG & Chatbot CT-Group 🚀

Tài liệu này ghi chú lại những cải tiến kỹ thuật đã được nhúng vào lõi hệ thống và vạch ra lộ trình nâng cấp thiết yếu chờ thực hiện khi giao diện người dùng (Frontend) đã sẵn sàng.

---

## ✅ Phần 1: Những thứ ĐÃ TỐI ƯU (Đang vận hành trong lõi)

### 1. Thuật toán Lắp Ráp Ngữ Cảnh Động (Dynamic Context Assembler)
- **Vị trí:** `app/services/generation.py`
- **Mô tả:** Hệ thống không bóp nghẹt `top_k` một cách cứng nhắc để phòng hờ các chunk bị cắt quá ngắn (gây mất thông tin). Thay vào đó, nó mở rộng tìm kiếm nhưng **sử dụng giới hạn trần ký tự (2500 chars)** để chủ động nạp/ngắt đoạn (chunk). Phương pháp này giúp não bộ LLM (như Qwen 1.7B) luôn bão hòa ở mức Context tốt nhất, vừa không bị đói thông tin, vừa đảm bảo giảm độ trễ (Latency) trả lời xuống sát ván mức < 15 giây.

### 2. Khiên chặn Trùng lặp Nội dung (Deduplication bằng SHA-256)
- **Vị trí:** `app/api/v1/extract.py` 
- **Mô tả:** Chặn người dùng nạp dồn cùng 1 tệp tin vào pipeline (dù đã cố ý đổi tên). File bị băm mã hóa SHA-256 thành "ADN" và đối chiếu với Lịch sử (`file_registry.json`). Nếu trùng, chặn tức khắc bằng lỗi HTTP 409, giúp tiết kiệm bộ nhớ Vector siêu sạch và triệt tiêu 100% việc phí phạm Compute GPU.

### 3. Sức bền của VLM OCR Phân Tán
- **Vị trí:** `app/api_paddle.py` & `pipeline/engines/paddle_engine.py` 
- **Mô tả:** 
  - Kích hoạt cơ chế Garbage Collection của CUDA qua biến `paddle.device.cuda.empty_cache()` để dọn sạch VRAM sau mỗi luồng ảnh OCR. Chống triệt để hiện tượng ngập lụt bộ nhớ (OOM) khi nạp file nhiều lần trên máy 4GB.
  - Ngưỡng giới hạn Timeout qua mạng nội bộ được nâng lên vùng cực đại (2_Tiếng / 7200s), sẵn sàng cân những tệp Nhân Sự PDF dài 50-100 trang mà không rớt mạng giữa chừng.

---

## ⏳ Phần 2: Những thứ CHƯA TỐI ƯU (Đợi Team Frontend Gắn UI)

### 1. Phản hồi Sinh Chữ Tức Thời (Server-Sent Events / Streaming)
- **Rào cản:** Frontend hiện tại đang bám dính vào cấu trúc JSON trả về ngầm định của `ChatResponse`. Thay đổi cấu trúc này sẽ làm gãy Code hiển thị của màn hình chat.
- **Hành động tương lai:** Khi UI Developer sẵn sàng bắt sự kiện chùm (Stream Chunk), ta sẽ thay lệnh `invoke` của LangChain bằng lệnh `stream()` và phát luồng Text trả về bằng `StreamingResponse` của FastAPI. Lúc này Bot sẽ gõ từng chữ một cách sinh động, "đánh lừa" cảm giác delay xuống mức 3-5 giây cho User.

### 2. Kiến trúc Máy chủ Inference Suy luận (vLLM)
- **Rào cản:** Hiện hệ thống đang chạy Docker & Local Windows, phải dùng Ollama Backend cho phần Text (Chat). Ollama khá ì ạch so với vLLM.
- **Hành động tương lai:** Tại Phase Đưa lên thực địa (Production Deployment) nếu sử dụng Dedicated Server Linux thuần, cần gỡ bỏ Ollama và Container-hóa (Dockerize) mô hình Qwen lên thẳng **vLLM Inference Server** (sử dụng *PagedAttention* và *Continuous Batching*). Thao tác này sẽ đẩy throughput tăng phi mã và triệt tiêu 80% độ gián đoạn trên quy mô hàng trăm Users đồng thời.
