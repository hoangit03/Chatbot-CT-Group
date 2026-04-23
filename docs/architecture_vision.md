# KẾ HOẠCH KIẾN TRÚC TỔNG THỂ (MASTER ARCHITECTURE)
**Dự Án:** Enterprise RAG Chatbot with RBAC & SharePoint Sync
**Mục tiêu:** Xây dựng hệ thống tự động đồng bộ tài liệu, chống tràn bộ nhớ đa phiên kết nối và bảo mật truy cập phân quyền theo phòng ban.

---

## 1. SƠ ĐỒ LUỒNG KIẾN TRÚC (MERMAID DIAGRAM)
*Hãy mở tệp này bằng bộ đọc Markdown trên VSCode/Obsidian hoặc dán lên GitHub để thấy toàn cảnh rực rỡ của sơ đồ.*

```mermaid
flowchart TD
    %% --------------------------------
    %% TẦNG GIAO TIẾP NGƯỜI DÙNG (FRONTEND)
    %% --------------------------------
    U_HR["Nhân sự HR"]
    U_FIN["Nhân sự Kế Toán"] 
    UI["Giao diện RAG Chatbot (Xác thực với M365)"]
    
    U_HR -->|Câu hỏi| UI
    U_FIN -->|Câu hỏi| UI

    %% --------------------------------
    %% TẦNG LƯU TRỮ DỮ LIỆU GỐC (DATA LAKE)
    %% --------------------------------
    SP["Mạng nội bộ: Microsoft SharePoint"]
    subgraph FolderTree [Cây Thư Mục Phân Quyền]
        SP_HR["📁 HR Internal (Mật)"]
        SP_FIN["📁 Tài chính (Mật)"]
        SP_PUB["📁 Chính sách chung (Public)"]
    end
    
    SP -.-> FolderTree
    
    %% --------------------------------
    %% TẦNG ĐỒNG BỘ & ĐIỀU HƯỚNG TẢI (INGESTION & QUEUE)
    %% --------------------------------
    Sync["Sync Worker (CronJob + MS Graph API)"]
    MQ{"Hàng đợi Message Queue (Redis / RabbitMQ)"}
    
    SP_HR -->|Cập nhật API| Sync
    SP_FIN -->|Cập nhật API| Sync
    SP_PUB -->|Cập nhật API| Sync
    
    Sync -->|Ném Task: doc_id, filepath| MQ
    
    %% --------------------------------
    %% TẦNG XỬ LÝ & ĐỊNH TUYẾN DỮ LIỆU (CORE PIPELINE)
    %% --------------------------------
    subgraph DataProcessing [Data Processing Pipeline]
        OCR["Trạm Trích xuất OCR \n Paddle Structure V3"]
        Router{"Trạm Phân Loại \n (Auto-Router)"}
        Embed["Trạm Nhúng Full-Text \n Dành cho Chính sách/Quy định"]
        MetaExtract["Trạm Siêu Dữ Liệu \n Dành cho Biểu Mẫu/Form"]
    end
    
    MQ -->|Pop từng File| OCR
    OCR -->|Đọc 10 dòng đầu để phân loại| Router
    
    Router -->|Nếu là Văn Bản, Quy Định| Embed
    Router -->|Nếu là Biểu Mẫu, Form| MetaExtract
    
    %% --------------------------------
    %% TẦNG CƠ SỞ DỮ LIỆU & RAG ENGINE
    %% --------------------------------
    VDB[(Vector Database: \n ChromaDB / Qdrant)]
    
    Embed -->|Lưu Vector Nội dung + Gắn nhãn \n dept: HR, doc_id: xxx| VDB
    MetaExtract -->|KHÔNG lưu nội dung. Kéo Tên Biểu Mẫu + Link tải SharePoint| VDB
    
    RAG["RAG Search Engine API \n LangChain / LlamaIndex"]
    LLM("Bot Local LLM \n Ollama Qwen2.5")
    
    %% --------------------------------
    %% LUỒNG TRUY VẤN CỦA CHATBOT (RAG FLOW)
    %% --------------------------------
    UI -->|1. Request Query + ID Phòng ban| RAG
    RAG -->|2. Ép Metadata Filter \n dept == UserDept, Public| VDB
    VDB -->|3. Trả về Văn cảnh Semantic \n Cấu trúc HTML Bảng giữ nguyên| RAG
    RAG -->|4. Lồng Context vào Prompt| LLM
    LLM -->|5. Đọc & Trả lời nhạy bén| RAG
    RAG -->|6. Phản hồi User| UI

    %% Styling
    classDef storage fill:#353b48,stroke:#fbc531,stroke-width:2px,color:#fff;
    classDef process fill:#0097e6,stroke:#273c75,stroke-width:2px,color:#fff;
    classDef worker fill:#44bd32,stroke:#2f3640,stroke-width:2px,color:#fff;
    classDef queue fill:#e1b12c,stroke:#e1b12c,stroke-width:2px,color:#fff;
    
    class SP,VDB storage;
    class RAG,LLM,UI process;
    class Sync,OCR,Embed worker;
    class MQ queue;
```

---

## 2. DIỄN GIẢI CHỨC NĂNG CÁC TRẠM

### Trạm Đệm Giảm Tải (Message Queue - Màu Vàng)
Đây là "người hùng" sửa lỗi sập hệ thống (OOM). Khi ném hàng trăm folder vào, Worker tạo ra 1000 Tasks. Thay vì PaddleOCR ép bản thân phải xử lý cùng 1000 cái (dẫn đến sập Cuda/VRAM), Queue bắt OCR phải làm theo thứ tự xếp hàng (Pop/Push). Giảm tải áp lực hệ thống trạm Local PC hoàn hảo.

### Trạm Cắt & Gắn Thẻ Phân Quyền (Embedding Worker)
Mọi Chunk text đi ra từ Markdown của Paddle đều không được nạp "trần trụi" vào ChromaDB. Chúng phải được **đóng dấu giáp lai (Metadata)**.
Ví dụ: `chunk_text: "Quy định mức Thưởng tết...", metadata: {"department": "HR", "access": "Confidential", "doc_id": "File_01"}`

### Trạm Truy vấn An toàn (RAG Search Engine)
Hệ API này sẽ can thiệp vào tầng Tìm Kiếm. 
Nếu ông A thuộc phòng Kế Toán tra cứu: Câu lệnh Vector DB sẽ bị "kẹp gông" một Filter: Tìm ý nghĩa tương đồng NHƯNG `metadata.department` BẮT BUỘC bằng `"KeToan"` hoặc `"Public"`.
Nó vĩnh viễn khóa đường vào của phòng Nhân Sự đối với ông Kế Toán này. Vừa giữ bảo mật đa phòng ban, vừa rút ngắn vòng quét Semantic.

## 3. KHUYẾN NGHỊ DEPLOYMENT DOCKER-COMPOSE
Trong Docker Compose tương lai của bạn, sẽ có 4 khối Services sau (Gắn chung 1 Bridge Network):
1. Khối `paddle_ocr`: Core như mình đang làm (Mở API chạy ngầm).
2. Khối `rag_server`: Bọc Source Code tìm kiếm API, Authentication, và kết nối Sharepoint.
3. Khối `vector_db`: Container chạy ChromaDB/Qdrant. 
4. Khối `redis`: Làm bãi đáp Message Queue xếp hàng nộp tài liệu.
