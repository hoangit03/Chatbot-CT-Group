import os
import re
import json
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

# === Path Resolution từ .env ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))

# Chunking config từ .env
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 250))

# MD files should be inside data_output/result
MD_DIR = os.path.join(SHARED_DIR, "data_output", "result")
# We will save chunks as json to a new directory
CHUNKS_DIR = os.path.join(SHARED_DIR, "data_output", "result_chunks")

# ── Template/Form markers: chunks chỉ chứa header metadata, không có nội dung ──
_TEMPLATE_MARKERS = [
    "Biểu mẫu / Form / Template",
    "Category: Biểu mẫu",
    "Cần điền tay",
]

# ── Custom separators: ưu tiên cắt tại RANH GIỚI MỤC, không cắt giữa nội dung ──
# Thứ tự: đoạn văn > numbered list > bullet > sub-bullet > dòng > câu > từ
_VIETNAMESE_SEPARATORS = [
    "\n\n",           # Đoạn văn (ưu tiên cao nhất)
    "\n1. ",           # Numbered list đầu mục
    "\n2. ",
    "\n3. ",
    "\n4. ",
    "\n5. ",
    "\n6. ",
    "\n7. ",
    "\n8. ",
    "\n9. ",
    "\n* ",            # Bullet points
    "\n+ ",            # Sub-bullets
    "\n- ",            # Dash bullets
    "\n",              # Dòng mới
    ". ",              # Câu
    " ",               # Từ
    "",                # Ký tự (fallback cuối)
]


def _is_template_chunk(content: str) -> bool:
    """Kiểm tra chunk có phải chỉ chứa template/form header không."""
    first_200 = content[:200]
    return any(marker in first_200 for marker in _TEMPLATE_MARKERS)


def _is_empty_chunk(content: str) -> bool:
    """Kiểm tra chunk rỗng hoặc chỉ chứa hình ảnh/link."""
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Bỏ ảnh markdown
    cleaned = re.sub(r'\[.*?\]\(.*?\)', '', cleaned)   # Bỏ link markdown
    cleaned = cleaned.strip()
    return len(cleaned) < 20


def run_chunking_pipeline(md_file_name: str) -> bool:
    """
    Tiến hành Băm MD File dựa trên Headers & Chunking Size

    Pipeline:
      1. MarkdownHeaderTextSplitter — cắt theo heading (#, ##, ###, ####)
      2. RecursiveCharacterTextSplitter — cắt nhỏ nếu section > CHUNK_SIZE
         Sử dụng custom separators ưu tiên ranh giới mục tiếng Việt
      3. Filter — loại bỏ chunks template/form và chunks rỗng
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
    print(f"[Chunking Engine] Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    print(f"[Chunking Engine] File size: {len(markdown_text)} chars")

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
    print(f"[Chunking Engine] Bước 1: Cắt theo heading → {len(md_header_splits)} sections")
    
    # Bước 2: Recursive Character Splitter với custom separators tiếng Việt
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_VIETNAMESE_SEPARATORS,
    )
    final_splits = text_splitter.split_documents(md_header_splits)
    print(f"[Chunking Engine] Bước 2: Recursive split → {len(final_splits)} chunks (trước filter)")

    # Bước 2.5: Contextual Headers — gắn heading section cha vào chunk con
    # Khi 1 section dài bị cắt thành N chunks, chunk 2..N mất heading
    # → Reranker không biết chunk đó thuộc section nào → score thấp → bị bỏ
    # Fix: Tìm heading cuối cùng trong mỗi chunk, gắn vào chunk tiếp theo
    _HEADING_PATTERN = re.compile(
        r'^(?:'
        r'#{1,4}\s+.+'           # Markdown heading: # ... #### ...
        r'|\d+\.\s+.{5,}'       # Numbered heading: 1. Tiêu đề dài
        r'|CHƯƠNG\s+[IVXLCDM0-9]+'  # CHƯƠNG I, CHƯƠNG II...
        r')',
        re.MULTILINE
    )
    
    current_last_heading = None
    contextual_injected = 0
    for i in range(len(final_splits)):
        content = final_splits[i].page_content
        
        # Tìm heading cuối cùng trong chunk này
        headings = _HEADING_PATTERN.findall(content)
        if headings:
            current_last_heading = headings[-1].strip()
            
        # Gắn heading này vào chunk TIẾP THEO nếu chunk đó không bắt đầu bằng heading
        if current_last_heading and (i + 1 < len(final_splits)):
            next_content = final_splits[i + 1].page_content
            if not _HEADING_PATTERN.match(next_content):
                # Prepend contextual header
                ctx_prefix = f"[Thuộc mục: {current_last_heading}]\n"
                final_splits[i + 1].page_content = ctx_prefix + next_content
                contextual_injected += 1

    if contextual_injected > 0:
        print(f"[Chunking Engine] Bước 2.5: Contextual Headers → gắn heading cho {contextual_injected} chunks")

    # Bước 3: Filter — loại bỏ template chunks và chunks rỗng
    filtered_template = 0
    filtered_empty = 0
    extracted_chunks = []

    for i, doc in enumerate(final_splits):
        content = doc.page_content

        if _is_template_chunk(content):
            filtered_template += 1
            continue

        if _is_empty_chunk(content):
            filtered_empty += 1
            continue

        extracted_chunks.append({
            "chunk_id": f"{md_file_name}_chunk_{len(extracted_chunks)}",
            "metadata": doc.metadata,
            "content": content
        })

    if filtered_template > 0 or filtered_empty > 0:
        print(f"[Chunking Engine] Bước 3: Filter → loại {filtered_template} template, {filtered_empty} rỗng")
        
    # Lưu ra ổ cứng
    base_name = os.path.splitext(md_file_name)[0]
    output_json_path = os.path.join(CHUNKS_DIR, f"{base_name}_chunks.json")
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_chunks, f, ensure_ascii=False, indent=4)
        
    print(f"[Chunking Engine] Thành công! {len(extracted_chunks)} chunks (từ {len(final_splits)} trước filter)")
    print(f"[Chunking Engine] Dữ liệu chunk được lưu tại: {output_json_path}")
    
    return True

