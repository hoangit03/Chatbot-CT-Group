"""
Cleaning Engine – Làm sạch file .md sau khi parse xong (OCR + ToMD).
Áp dụng cho toàn bộ .md output trước khi đưa sang Chunking Worker.

Các bước làm sạch:
  1. Xóa filler patterns TRONG dòng (....  …  ···  ‧‧‧  __  **)
  2. Xóa dòng separator dash-only (|---|---|---|)
  3. Chuẩn hóa khoảng trắng
  4. Xóa dòng trống / chỉ ký tự đặc biệt
  5. Chuẩn hóa nhiều dòng trống liên tiếp → tối đa 1 dòng trống

Gợi ý bổ sung (có thể bật tắt):
  6. [OPTION] Normalize dấu câu tiếng Việt (full-width → half-width)
  7. [OPTION] Strip HTML entity còn sót (&amp; &nbsp; v.v.)
  8. [OPTION] Xóa page-number pattern ("Trang X/Y" hoặc "Page X of Y")
"""
import os
import re
from dotenv import load_dotenv

load_dotenv()

# === Path Resolution từ .env ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))

# Vị trí chứa các .md đầu ra (cả từ OCR lẫn ToMD đều đổ về đây)
MD_OUTPUT_DIR = os.path.join(SHARED_DIR, "data_output", "result")

# ==========================================
# REGEX PATTERNS
# ==========================================

# Filler (điền tay, dòng chừa chỗ): xóa khỏi TRONG dòng
FILLER_PATTERNS = [
    r"\.{4,}",      # ....
    r"…+",          # … (unicode ellipsis)
    r"·{3,}",       # ···
    r"‧{3,}",       # ‧‧‧ (dot leader)
    r"_{2,}",       # __
    r"\*{2,}",      # **
]
FILLER_REGEX = re.compile("|".join(FILLER_PATTERNS))

# Dòng chỉ có ký tự đặc biệt (không phải chữ cái, số, dấu tiếng Việt)
SPECIAL_LINE_RE = re.compile(r"^[^\wÀ-ỹ]+$")

# Dòng separator markdown table hoặc dash separator: |---|---|, ---, ===
DASH_SEP_RE = re.compile(r"^[\|\-\s=:]+$")

# HTML entities còn sót
HTML_ENTITY_RE = re.compile(r"&\w+;")

# Page number pattern: "Trang 1/5", "Page 2 of 10"
PAGE_NUM_RE = re.compile(r"\b(Trang|Page)\s+\d+\s*(\/|of)\s*\d+\b", re.IGNORECASE)

# Full-width punctuation → half-width
FULLWIDTH_MAP = str.maketrans(
    "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)


# ==========================================
# CORE CLEANING FUNCTION
# ==========================================
def clean_md_text(text: str,
                  strip_html_entities: bool = True,
                  strip_page_numbers: bool = True,
                  normalize_fullwidth: bool = True) -> str:
    """
    Làm sạch toàn bộ nội dung text của một file .md.
    Trả về chuỗi đã được làm sạch.
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # ---- Bước 1: xóa HTML entities ----
        if strip_html_entities:
            line = HTML_ENTITY_RE.sub(" ", line)

        # ---- Bước 2: xóa số trang ----
        if strip_page_numbers:
            line = PAGE_NUM_RE.sub("", line)

        # ---- Bước 3: bỏ qua dòng dash separator (|---|---|) ----
        stripped = line.strip()
        if DASH_SEP_RE.fullmatch(stripped):
            continue

        # ---- Bước 4: xóa filler TRONG dòng ----
        line = FILLER_REGEX.sub("", line)

        # ---- Bước 5: normalize full-width punctuation ----
        if normalize_fullwidth:
            line = line.translate(FULLWIDTH_MAP)

        # ---- Bước 6: chuẩn hóa khoảng trắng ----
        line = " ".join(line.split())

        # ---- Bước 7: bỏ dòng trống hoặc chỉ ký tự đặc biệt ----
        if not line:
            cleaned_lines.append("")  # giữ 1 dòng trống làm phân cách
            continue
        if SPECIAL_LINE_RE.fullmatch(line):
            continue

        cleaned_lines.append(line)

    # ---- Bước 8: thu gọn nhiều dòng trống liên tiếp → tối đa 1 ----
    result_lines = []
    prev_empty = False
    for line in cleaned_lines:
        if line == "":
            if not prev_empty:
                result_lines.append("")
            prev_empty = True
        else:
            result_lines.append(line)
            prev_empty = False

    return "\n".join(result_lines).strip()


# ==========================================
# PIPELINE FUNCTION (xử lý 1 file)
# ==========================================
def run_cleaning_pipeline(md_file_name: str) -> bool:
    """
    Đọc file .md trong OUTPUT_DIR, làm sạch và ghi đè lại.
    Trả về True nếu thành công.
    """
    file_path = os.path.join(MD_OUTPUT_DIR, md_file_name)

    if not os.path.exists(file_path):
        print(f"[Cleaning Engine] File không tồn tại: {file_path}")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_text = f.read()

        cleaned_text = clean_md_text(original_text)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        original_lines = len(original_text.splitlines())
        cleaned_lines = len(cleaned_text.splitlines())
        reduction = original_lines - cleaned_lines
        print(f"[Cleaning Engine] Done: {md_file_name} | {original_lines} -> {cleaned_lines} dòng (-{reduction})")
        return True

    except Exception as e:
        print(f"[Cleaning Engine] Lỗi khi làm sạch {md_file_name}: {e}")
        return False
