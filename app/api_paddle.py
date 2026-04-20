from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import os
import uuid
import fitz  # PyMuPDF
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
            
        print(f"[VLM Engine] Bắt đầu trích xuất luồng file {req.file_name}...", flush=True)
        final_markdown = []
        
        # Hàm xử lý chung kết quả VLM
        def process_page_res(page_res):
            print(f"[DEBUG] process_page_res Type: {type(page_res)}", flush=True)
            if isinstance(page_res, dict):
                print(f"[DEBUG] Keys: {list(page_res.keys())}", flush=True)
            else:
                print(f"[DEBUG] Attrs: {dir(page_res)}", flush=True)
                if hasattr(page_res, '__dict__'):
                    print(f"[DEBUG] Dict vars: {page_res.__dict__.keys()}", flush=True)
                    
            page_md = []
            if isinstance(page_res, dict):
                print(f"[DEBUG] Has parsing_res_list: {'parsing_res_list' in page_res}", flush=True)
                if 'parsing_res_list' in page_res:
                    print(f"[DEBUG] parsing_res_list len: {len(page_res['parsing_res_list'])}", flush=True)
                    if len(page_res['parsing_res_list']) > 0:
                        first = page_res['parsing_res_list'][0]
                        print(f"[DEBUG] First block type: {type(first)}, dict: {first if isinstance(first, dict) else dir(first)}", flush=True)
                if 'markdown' in page_res and isinstance(page_res['markdown'], dict) and 'text' in page_res['markdown']:
                    page_md.append(page_res['markdown']['text'])
                elif 'parsing_res_list' in page_res:
                    parsing_res = page_res.get("parsing_res_list", [])
                    def _get_val(obj, key, default=None):
                        if isinstance(obj, dict):
                            return obj.get(key, default)
                        return getattr(obj, key, default)
                    
                    # Sort if block_order exists
                    parsing_res = sorted(parsing_res, key=lambda x: _get_val(x, 'block_order') if _get_val(x, 'block_order') is not None else 9999)
                    
                    for block in parsing_res:
                        # Support PPStructureV3 AND PaddleOCR-VL-1.5 formats
                        label = _get_val(block, 'label') or _get_val(block, 'block_label')
                        orig_text = _get_val(block, 'content') or _get_val(block, 'text') or _get_val(block, 'block_content', '')
                        orig_text = str(orig_text).strip() if orig_text else ''
                        if not orig_text:
                            continue
                            
                        # Format mapping
                        if label in ['table', 'Table']:
                            page_md.append(f"\n{orig_text}\n")
                        elif label in ['doc_title', 'Title']:
                            page_md.append(f"# {orig_text}\n")
                        elif label in ['header', 'Header', 'title']:
                            page_md.append(f"**{orig_text}**\n")
                        elif label in ['paragraph_title', 'Paragraph']:
                            page_md.append(f"### {orig_text}\n")
                        else:
                            page_md.append(f"{orig_text}\n\n")
                elif 'chat_res' in page_res:
                    page_md.append(str(page_res.get('chat_res', '')))
                elif 'raw_out' in page_res:
                    page_md.append(str(page_res.get('raw_out', '')))
                elif 'res' in page_res and 'markdown' in page_res['res']: # sometimes nested
                    page_md.append(str(page_res['res']['markdown']))
            else:
                if hasattr(page_res, 'markdown'):
                    md_val = page_res.markdown
                    if isinstance(md_val, str):
                        page_md.append(md_val)
                    elif hasattr(md_val, '__repr__'):
                        page_md.append(str(md_val))
                elif hasattr(page_res, 'chat_res'):
                    page_md.append(str(page_res.chat_res))
                elif hasattr(page_res, '__dict__') and 'chat_res' in page_res.__dict__:
                    page_md.append(str(page_res.__dict__.get('chat_res', '')))
                elif hasattr(page_res, 'get'):
                    # Call nested dict
                    doc_dict = getattr(page_res, 'get')('res', {})
                    if 'markdown' in doc_dict:
                        page_md.append(str(doc_dict['markdown']))
                    elif 'raw_out' in doc_dict:
                        page_md.append(str(doc_dict['raw_out']))
                        
            return "\n".join(page_md)

        # 2. Xử lý từng trang PDF bằng cách tách thành Image (Giảm tải VRAM OOM)
        if ext.lower() == '.pdf':
            doc = fitz.open(temp_path)
            print(f"[VLM Engine] Tách PDF thành {len(doc)} ảnh để bảo vệ VRAM", flush=True)
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                # Tăng DPi resolution để OCR rõ nét nhưng không quá lớn
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_path = f"/data/input/{temp_id}_page{page_idx}.png"
                pix.save(img_path)
                
                print(f"[VLM Engine] Analyzing Page {page_idx}", flush=True)
                # Đưa từng ảnh vào qua pipeline (1 in 1 out)
                img_result = pipeline(img_path)
                
                # Quét qua result stream của 1 page này
                for item in img_result:
                    parsed_text = process_page_res(item)
                    final_markdown.append(parsed_text)
                    
                # Xóa ngay lập tức ảnh trên đĩa để giải phóng
                os.remove(img_path)
                
            doc.close()
        else:
            # Dành cho các định dạng không phải PDF (VD: png, jpg)
            result = pipeline(temp_path)
            for idx, page_res in enumerate(result):
                print(f"[VLM Engine] Analyzing Component {idx}", flush=True)
                parsed_text = process_page_res(page_res)
                final_markdown.append(parsed_text)
        
        md_text = "\n\n---\n\n".join(final_markdown)
        print(f"[VLM Engine] Hoàn tất {req.file_name}! Thu được {len(md_text)} char.", flush=True)
        
        return OCRResponse(markdown=md_text, success=True)
        
    except Exception as e:
        import traceback
        err_msg = str(e) + "\n" + traceback.format_exc()
        print(f"[VLM Engine] Sập ở file {req.file_name}: {err_msg}", flush=True)
        return OCRResponse(markdown="", success=False, error=err_msg)
        
    finally:
        # Xóa file rác tiết kiệm ổ cứng cho máy 4GB
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
