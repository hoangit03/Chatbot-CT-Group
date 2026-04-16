import os
import sys
import glob
import pandas as pd
import mammoth
import markdownify
import extract_msg
import subprocess
import shutil
import re
from dotenv import load_dotenv

load_dotenv()

# === Path Resolution từ .env ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SHARED_DIR = os.getenv("SHARED_DATA_DIR", os.path.join(REPO_ROOT, "shared_data"))

INPUT_DIR = os.path.join(SHARED_DIR, "data_input")
OUTPUT_BASE = os.path.join(SHARED_DIR, "data_output")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "result")
TMP_DIR = os.path.join(OUTPUT_BASE, "tmp_conversion")
IMG_DIR = os.path.join(TMP_DIR, "images_from_md")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import broker publisher (clean import thay vì sys.path hack)
from app.services.broker_service import ocr_publisher

# ==========================================
# MODULE 1: FORM DETECTION
# ==========================================
def classify_markdown(md_text: str) -> str:
    """Nhận diện biểu mẫu/form qua regex filler patterns, gắn cờ lên đầu file."""
    filler_patterns = [
        r"\.{4,}",
        r"…{1,}",
        r"·{4,}",
        r"‧{4,}",
        r"-{4,}",
        r"_{4,}",
    ]
    combined_pattern = re.compile("|".join(filler_patterns))
    if combined_pattern.search(md_text):
        return "> [!NOTE]\n> **Category:** Biểu mẫu / Form / Template (Cần điền tay)\n\n"
    else:
        return "> [!NOTE]\n> **Category:** Quy định / Chính sách / Thông báo\n\n"

# ==========================================
# MODULE 2: PPT TO PDF (Windows only)
# ==========================================
def convert_ppt_xplatform(input_path: str, output_dir: str) -> str:
    if sys.platform != 'win32':
        print("[ToMD Engine] PPT convert chỉ hỗ trợ Windows.")
        return None
    import win32com.client
    safe_temp_in = os.path.join(output_dir, "safe_temp_convert_ppt.pptx")
    safe_temp_out = os.path.join(output_dir, "safe_temp_convert_ppt.pdf")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    final_output_path = os.path.join(output_dir, base_name + ".pdf")
    try:
        shutil.copy2(input_path, safe_temp_in)
        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        powerpoint.Visible = 1
        presentation = powerpoint.Presentations.Open(os.path.abspath(safe_temp_in), WithWindow=False)
        presentation.SaveAs(os.path.abspath(safe_temp_out), 32)
        presentation.Close()
        powerpoint.Quit()
        if os.path.exists(safe_temp_out):
            shutil.move(safe_temp_out, final_output_path)
        try: os.remove(safe_temp_in)
        except: pass
        print(f"[win32com] PPT -> PDF: {final_output_path}")
        return final_output_path
    except Exception as e:
        print(f"[Engine Error] PPT Convert Fail: {e}")
        return None

# ==========================================
# MODULE 3: DOC -> DOCX (Cross-platform)
# ==========================================
def convert_doc_xplatform(input_path: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, base_name + ".docx")
    if sys.platform == 'win32':
        import win32com.client
        try:
            safe_temp_in = os.path.join(output_dir, "safe_temp_convert_wd.doc")
            safe_temp_out = os.path.join(output_dir, "safe_temp_convert_wd.docx")
            shutil.copy2(input_path, safe_temp_in)
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0
            doc = word.Documents.Open(os.path.abspath(safe_temp_in))
            doc.SaveAs(os.path.abspath(safe_temp_out), FileFormat=16)
            doc.Close(False)
            word.Quit()
            if os.path.exists(safe_temp_out):
                shutil.move(safe_temp_out, output_path)
            try: os.remove(safe_temp_in)
            except: pass
            print(f"[win32com] DOC -> DOCX: {output_path}")
            return output_path
        except Exception as e:
            print(f"[Engine Error] DOC Convert Fail: {e}")
            return None
    else:
        try:
            cmd = ["libreoffice", "--headless", "--convert-to", "docx", input_path, "--outdir", output_dir]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[libreoffice] DOC -> DOCX: {output_path}")
            return output_path
        except Exception as e:
            print(f"[Engine Error] LibreOffice DOC Fail: {e}")
            return None

# ==========================================
# MODULE 4: DOCX -> MD
# ==========================================
def docx_to_md(input_path, output_path, img_root_dir):
    try:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        img_folder = os.path.join(img_root_dir, base_name)
        os.makedirs(img_folder, exist_ok=True)
        img_counter = {"count": 0}

        def save_image(image):
            img_counter["count"] += 1
            ext = image.content_type.split("/")[-1]
            img_path = os.path.join(img_folder, f"img_{img_counter['count']}.{ext}")
            with image.open() as img_bytes:
                with open(img_path, "wb") as f:
                    f.write(img_bytes.read())
            return {"src": os.path.relpath(img_path, os.path.dirname(output_path))}

        with open(input_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file, convert_image=mammoth.images.img_element(save_image))
            html = result.value

        md = markdownify.markdownify(html)
        md = re.sub(r'\n\s*\n+', '\n\n', md)
        category_flag = classify_markdown(md)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(category_flag + md)
        print(f"DOCX -> MD: {output_path}")
    except Exception as e:
        print(f"[Engine Error] DOCX -> MD Fail: {e}")

# ==========================================
# MODULE 5: EXCEL -> MD (v2 - Advanced Parsing)
# ==========================================
def excel_to_md(input_path, output_path):
    try:
        def normalize_df(df):
            df = df.astype(str)
            df = df.replace(["nan", "NaN", "None"], "")
            df = df.replace(r'^\s*$', "", regex=True)
            return df

        def split_blocks_v2(df, min_empty_rows=2):
            """V2: fix merged cells bằng ffill, soft threshold tránh split vụn."""
            df = df.copy()
            df = df.fillna("")
            df = df.replace(r'^\s*$', "", regex=True)
            df = df.ffill(axis=0)  # fix merged cells
            df = df.ffill(axis=1)

            row_blocks, current, empty_count = [], [], 0
            for _, row in df.iterrows():
                if all(str(v).strip() == "" for v in row):
                    empty_count += 1
                else:
                    if empty_count >= min_empty_rows and current:
                        row_blocks.append(pd.DataFrame(current))
                        current = []
                    empty_count = 0
                    current.append(row)
            if current:
                row_blocks.append(pd.DataFrame(current))

            def refine_by_width(block):
                sub_blocks, current, prev_width = [], [], None
                for _, row in block.iterrows():
                    width = sum(1 for v in row if str(v).strip() != "")
                    if prev_width is not None and width > prev_width * 1.5:
                        if current:
                            sub_blocks.append(pd.DataFrame(current))
                            current = []
                    current.append(row)
                    prev_width = width
                if current:
                    sub_blocks.append(pd.DataFrame(current))
                return sub_blocks

            def split_by_columns_soft(block, empty_ratio=0.95):
                sub_blocks, current_cols = [], []
                for i in range(block.shape[1]):
                    col = block.iloc[:, i]
                    empty = sum(str(v).strip() == "" for v in col) / len(col)
                    if empty < empty_ratio:
                        current_cols.append(i)
                    else:
                        if current_cols:
                            sub_blocks.append(block.iloc[:, current_cols])
                            current_cols = []
                if current_cols:
                    sub_blocks.append(block.iloc[:, current_cols])
                return sub_blocks

            final_blocks = []
            for block in row_blocks:
                for r_block in refine_by_width(block):
                    for sub in split_by_columns_soft(r_block):
                        sub = sub.loc[(sub != "").any(axis=1)]
                        if not sub.empty:
                            final_blocks.append(sub.reset_index(drop=True))
            return final_blocks

        def is_table(block):
            if block.empty: return False
            block = block.loc[(block != "").any(axis=1), :]
            block = block.loc[:, (block != "").any(axis=0)]
            rows, cols = block.shape
            return (rows >= 2 and cols >= 2) or cols >= 4

        def clean_table(block):
            block = block.copy()
            block = block.loc[(block != "").any(axis=1), :]
            block = block.loc[:, (block != "").any(axis=0)]
            if block.empty: return block
            block = block.reset_index(drop=True)
            header = block.iloc[0]
            if (header != "").sum() >= 2:
                block.columns = header
                block = block[1:].reset_index(drop=True)
            else:
                block.columns = [f"col_{i}" for i in range(block.shape[1])]
            return block

        def block_to_text(block):
            block = block.fillna("")
            lines = []
            for _, row in block.iterrows():
                vals = [str(v).strip() for v in row if str(v).strip()]
                if vals:
                    lines.append(" ".join(" ".join(vals).split()))
            return "\n".join(lines)

        xls = pd.ExcelFile(input_path)
        all_md = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            df = normalize_df(df)
            if df.empty or (df == "").all().all(): continue
            blocks = split_blocks_v2(df)
            sheet_md = [f"# Sheet: {sheet_name}"]
            for block in blocks:
                if block.empty: continue
                if is_table(block):
                    table = clean_table(block)
                    if not table.empty:
                        try: sheet_md.append(table.to_markdown(index=False))
                        except: sheet_md.append(table.to_string(index=False))
                else:
                    text = block_to_text(block)
                    if text.strip(): sheet_md.append(text)
            all_md.append("\n\n".join(sheet_md))

        final_md_text = "\n\n".join(all_md)
        category_flag = classify_markdown(final_md_text)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(category_flag + final_md_text)
        print(f"Excel -> MD (v2): {output_path}")
    except Exception as e:
        print(f"[Engine Error] Excel -> MD Fail: {e}")

# ==========================================
# MODULE 6: MSG EMAIL -> MD
# ==========================================
def msg_to_md(input_path, output_path):
    try:
        msg = extract_msg.Message(input_path)
        md_parts = [
            f"# Subject: {msg.subject or ''}",
            f"**From:** {msg.sender or ''}",
            f"**Date:** {msg.date or ''}\n",
            "## Body\n",
            msg.body or ""
        ]
        final_md_text = "\n".join(md_parts)
        category_flag = classify_markdown(final_md_text)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(category_flag + final_md_text)
        print(f"MSG -> MD: {output_path}")
    except Exception as e:
        print(f"[Engine Error] MSG -> MD Fail: {e}")

# ==========================================
# MODULE 7: MAIN PIPELINE DISPATCHER
# ==========================================
def run_to_md_pipeline(file_name: str) -> bool:
    input_path = os.path.join(INPUT_DIR, file_name)
    base_name, ext = os.path.splitext(file_name)
    ext = ext.lower().replace('.', '')
    output_path = os.path.join(OUTPUT_DIR, base_name + ".md")

    print(f"\n[ToMD Engine] Processing: {file_name} (ext={ext})")

    if ext == 'doc':
        new_path = convert_doc_xplatform(input_path, TMP_DIR)
        if new_path:
            docx_to_md(new_path, output_path, IMG_DIR)
        else:
            return False
    elif ext == 'docx':
        docx_to_md(input_path, output_path, IMG_DIR)
    elif ext in ['xls', 'xlsx']:
        excel_to_md(input_path, output_path)
    elif ext == 'msg':
        msg_to_md(input_path, output_path)
    elif ext in ['ppt', 'pptx']:
        pdf_path = convert_ppt_xplatform(input_path, TMP_DIR)
        if pdf_path:
            pdf_name = os.path.basename(pdf_path)
            final_ocr_path = os.path.join(INPUT_DIR, pdf_name)
            shutil.copy2(pdf_path, final_ocr_path)
            print(f"[ToMD Engine] PPT->PDF done. Pushing to OCR queue: {pdf_name}")
            ocr_publisher.publish_task(pdf_name)
            return True
        else:
            return False
    else:
        print(f"[ToMD Engine] Unsupported format or skip: {ext}")
        return False

    print(f"[ToMD Engine] Done: {file_name}")
    return True
