import os
import subprocess
import json
import glob
import shutil
from dotenv import load_dotenv

load_dotenv()

# === Path Resolution từ .env (thống nhất toàn hệ thống) ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))

INPUT_DIR = os.path.join(SHARED_DIR, "data_input")
OUTPUT_DIR = os.path.join(SHARED_DIR, "data_output")
RESULT_DIR = os.path.join(OUTPUT_DIR, "result")

def run_ocr_pipeline(pdf_name: str) -> bool:
    """
    Ham Chuc Nang Dong Goi Core Logic
    Thuc thi viec day xuong Docker -> nhan JSON -> xuat Markdown
    """
    base_name = os.path.splitext(pdf_name)[0]
    print(f"\n[Paddle Engine Loi] Bat dau kich hoat lenh Paddle OCR cho file: {pdf_name}", flush=True)
    
    docker_input_path = f"/data/input/{pdf_name}"
    cmd = [
        "docker", "exec", "paddleocr_server", 
        "paddlex", "--pipeline", "PP-StructureV3", 
        "--input", docker_input_path, 
        "--device", "gpu", 
        "--save_path", "/data/output"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[Engine Error] PaddleX Docker ngung tre voi file {pdf_name}: {e}")
        return False
        
    json_candidates = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{base_name}*_res.json")))
    if not json_candidates:
        print(f"[Engine Error] Khong tim thay output JSON cho: {pdf_name}")
        return False
        
    final_markdown = []
    
    for json_path in json_candidates:
        with open(json_path, 'r', encoding='utf-8') as f:
            paddle_data = json.load(f)
            
        parsing_res = paddle_data.get("parsing_res_list", [])
        parsing_res = sorted(parsing_res, key=lambda x: x.get('block_order') if x.get('block_order') is not None else 9999)
        
        for block in parsing_res:
            label = block.get('block_label')
            orig_text = block.get('block_content', '').strip()
            
            if label == 'seal':
                final_markdown.append("\n[Image: Seal/Signature]\n")
                continue
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

    markdown_result = "\n".join(final_markdown)
    os.makedirs(RESULT_DIR, exist_ok=True)
    final_md_path = os.path.join(RESULT_DIR, f"{base_name}_paddle_only.md")
    
    with open(final_md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_result)

    print(f"[Paddle Engine Loi] Da duc thanh cong khoi Markdown cho {pdf_name}")

    # Thu don rac
    cleanup_files = glob.glob(os.path.join(OUTPUT_DIR, f"{base_name}*"))
    for f in cleanup_files:
        if os.path.isfile(f):
            try:
                os.remove(f)
            except:
                pass
    paddle_imgs_dir = os.path.join(OUTPUT_DIR, "imgs")
    if os.path.exists(paddle_imgs_dir):
        shutil.rmtree(paddle_imgs_dir, ignore_errors=True)
        
    print(f"[Paddle Engine Loi] Giao hang hoan thien!")
    return True
