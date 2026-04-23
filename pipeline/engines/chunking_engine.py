import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

# === Path Resolution từ .env ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))

# Chunking config từ .env
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# MD files should be inside data_output/result
MD_DIR = os.path.join(SHARED_DIR, "data_output", "result")
# We will save chunks as json to a new directory
CHUNKS_DIR = os.path.join(SHARED_DIR, "data_output", "result_chunks")

def run_chunking_pipeline(md_file_name: str) -> bool:
    """
    Tiến hành Băm MD File dựa trên Headers & Chunking Size
    """
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    md_path = os.path.join(MD_DIR, md_file_name)
    if not os.path.exists(md_path):
        # Fallback thử tìm ở thư mục data_output cha
        md_path = os.path.join(SHARED_DIR, "data_output", md_file_name)
        if not os.path.exists(md_path):
            print(f"[Chunking Engine] Lỗi: Không tìm thấy file MD {md_file_name}")
            return False

    print(f"\n[Chunking Engine] Bắt đầu băm file: {md_file_name}")
    
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # Bước 1: Băm theo Tiêu Đề
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)
    
    # Bước 2: Recursive Character Splitter cho các mảng văn bản quá to
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    final_splits = text_splitter.split_documents(md_header_splits)
    
    # Đóng gói và lưu JSON
    extracted_chunks = []
    for i, doc in enumerate(final_splits):
        extracted_chunks.append({
            "chunk_id": f"{md_file_name}_chunk_{i}",
            "metadata": doc.metadata,
            "content": doc.page_content
        })
        
    # Lưu ra ổ cứng
    base_name = os.path.splitext(md_file_name)[0]
    output_json_path = os.path.join(CHUNKS_DIR, f"{base_name}_chunks.json")
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_chunks, f, ensure_ascii=False, indent=4)
        
    print(f"[Chunking Engine] Thành công! Đã tạo ra {len(extracted_chunks)} chunks cực chuẩn.")
    print(f"[Chunking Engine] Dữ liệu chunk được lưu tại: {output_json_path}")
    
    return True
