from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import os
import uuid
from paddlex import create_pipeline

app = FastAPI(title="PaddleOCR-VL-1.5 Native Server", version="1.0")

# Khởi tạo mô hình siêu to khổng lồ VLM
print("[VLM Engine] Đang nạp PaddleOCR-VL-1.5 vào VRAM 4GB...")
try:
    pipeline = create_pipeline('PaddleOCR-VL-1.5')
    print("[VLM Engine] Nạp model thành công!")
except Exception as e:
    print(f"[VLM Engine Lỗi Cấp Bách] Không nạp được mô hình PaddleOCR-VL-1.5: {e}")
    # Fallback to structure v3 if VL doesn't exist
    try:
        pipeline = create_pipeline('PP-StructureV3')
        print("[VLM Engine] Nạp dự phòng PP-StructureV3 thành công!")
    except Exception as inner_e:
        pipeline = None

class OCRRequest(BaseModel):
    file_b64: str
    file_name: str = "temp.pdf"

class OCRResponse(BaseModel):
    markdown: str
    success: bool
    error: str = ""

@app.post("/layout-parsing", response_model=OCRResponse)
def predict_layout(req: OCRRequest):
    if pipeline is None:
        return OCRResponse(markdown="", success=False, error="VLM Pipeline chưa được khởi tạo!")

    # Đường dẫn file tạm trong Docker
    temp_id = str(uuid.uuid4())
    ext = os.path.splitext(req.file_name)[1]
    temp_path = f"/data/input/{temp_id}{ext}"
    
    try:
        # 1. Giải mã Base64 thành file vật lý
        file_bytes = base64.b64decode(req.file_b64)
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
            
        print(f"[VLM Engine] Bắt đầu trích xuất luồng file {req.file_name}...")
        
        # 2. Xử lý bằng Native SDK
        # Chạy model trên nội dung file
        result = pipeline(temp_path)
        
        # 3. Thu gom Markdown Native
        final_markdown = []
        for idx, page_res in enumerate(result):
            # page_res là một dictionary trả về bởi Pipeline.
            if 'markdown' in page_res and isinstance(page_res['markdown'], dict) and 'text' in page_res['markdown']:
                # Nếu API Mới có hỗ trợ 'markdown' (FastAPI chuẩn format)
                final_markdown.append(page_res['markdown']['text'])
            elif 'parsing_res_list' in page_res:
                # Nếu phiên bản Engine sâu bên trong trả ra List Block (Native SDK Thường thấy)
                parsing_res = page_res.get("parsing_res_list", [])
                parsing_res = sorted(parsing_res, key=lambda x: x.get('block_order') if x.get('block_order') is not None else 9999)
                
                for block in parsing_res:
                    label = block.get('block_label')
                    orig_text = block.get('block_content', '').strip()
                    if not orig_text:
                        continue
                    if label == 'table':
                        final_markdown.append(f"\n{orig_text}\n")
                    elif label == 'doc_title':
                        final_markdown.append(f"# {orig_text}\n")
                    elif label == 'header':
                        final_markdown.append(f"**{orig_text}**\n")
                    elif label == 'paragraph_title':
                        final_markdown.append(f"### {orig_text}\n")
                    else:
                        final_markdown.append(f"{orig_text}\n\n")
            elif 'chat_res' in page_res:
                # Cho cấu trúc trả về đặc thù của PaddleOCR-VL-1.5 (VLM Models)
                final_markdown.append(str(page_res.get('chat_res', '')))
            elif 'raw_out' in page_res:
                final_markdown.append(str(page_res.get('raw_out', '')))
        
        md_text = "\n".join(final_markdown)
        print(f"[VLM Engine] Hoàn tất {req.file_name}! Thu được {len(md_text)} char.")
        
        return OCRResponse(markdown=md_text, success=True)
        
    except Exception as e:
        import traceback
        err_msg = str(e) + "\n" + traceback.format_exc()
        print(f"[VLM Engine] Sập ở file {req.file_name}: {err_msg}")
        return OCRResponse(markdown="", success=False, error=str(e))
        
    finally:
        # Xóa file rác tiết kiệm ổ cứng cho máy 4GB
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
