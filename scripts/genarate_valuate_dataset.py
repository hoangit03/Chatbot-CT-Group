"""
generate_evaluate_dataset.py
Tạo dataset đánh giá RAG bằng HybridRetriever → xuất file Excel tương thích RAGAS.

FIX:
- Serialize contexts/sources thành JSON string (Excel không lưu list Python)
- Thêm cột RAGAS-ready: question, answer, contexts, ground_truth
- Tách cột contexts_str (JSON) và contexts_preview (dễ đọc khi review thủ công)
- Xử lý lỗi từng row, không crash toàn bộ batch
- In progress % rõ ràng
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from scripts.hybrid_retriever import HybridRetriever


def generate_evaluation_dataset_with_hybrid(
    input_excel: str = "data/evaluation/test_dataset.xlsx",
    output_excel: str = None,
    top_k: int = 8
):
    print("🚀 BẮT ĐẦU TẠO DATASET ĐÁNH GIÁ BẰNG HYBRID RETRIEVER...\n")

    df = pd.read_excel(input_excel)
    print(f"📊 Đã load {len(df)} câu hỏi từ: {input_excel}")

    # Validate cột bắt buộc
    if "question" not in df.columns:
        raise ValueError("File Excel phải có cột 'question'")

    # Khởi tạo HybridRetriever 1 lần duy nhất (không tạo lại trong loop)
    hybrid = HybridRetriever(top_k=top_k)

    try:
        hybrid._initialize()  
    except Exception as e:
        print(f"❌ Không thể khởi tạo HybridRetriever: {e}")
        raise 

    contexts_json_list = []       # JSON string → dùng cho RAGAS
    sources_json_list = []        # JSON string → metadata nguồn
    contexts_preview_list = []    # Plain text → dễ đọc khi review thủ công
    retrieved_count_list = []
    has_error_list = []

    total = len(df)
    for idx, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        if not question:
            print(f"[{idx+1:3d}/{total}] ⚠️  Bỏ qua row rỗng")
            contexts_json_list.append("[]")
            sources_json_list.append("[]")
            contexts_preview_list.append("")
            retrieved_count_list.append(0)
            has_error_list.append(True)
            continue

        print(f"[{idx+1:3d}/{total}] {question[:75]}...")

        try:
            docs = hybrid.retrieve(question, top_k=top_k)

            # FIX: serialize thành JSON string để Excel lưu được
            contexts = [doc.page_content for doc in docs]
            sources = [
                {
                    "file_name": doc.metadata.get("file_name", doc.metadata.get("source", "Unknown")),
                    "score": round(float(doc.metadata.get("similarity_score", 0.0)), 4)
                }
                for doc in docs
            ]

            contexts_json = json.dumps(contexts, ensure_ascii=False)
            sources_json = json.dumps(sources, ensure_ascii=False)

            # Preview ngắn gọn để review
            preview_lines = []
            for i, ctx in enumerate(contexts[:3]):  # chỉ preview 3 context đầu
                snippet = ctx[:200].replace("\n", " ")
                preview_lines.append(f"[{i+1}] {snippet}...")
            preview = "\n---\n".join(preview_lines)

            contexts_json_list.append(contexts_json)
            sources_json_list.append(sources_json)
            contexts_preview_list.append(preview)
            retrieved_count_list.append(len(docs))
            has_error_list.append(False)

        except Exception as e:
            print(f"   ❌ Lỗi: {type(e).__name__}: {e}")
            contexts_json_list.append("[]")
            sources_json_list.append("[]")
            contexts_preview_list.append(f"ERROR: {e}")
            retrieved_count_list.append(0)
            has_error_list.append(True)

    # ── Gắn các cột mới vào DataFrame ──────────────────────────────────────
    df["contexts"] = contexts_json_list          # JSON list → dùng cho RAGAS
    df["sources"] = sources_json_list            # JSON list metadata
    df["contexts_preview"] = contexts_preview_list  # Dễ đọc khi review thủ công
    df["retrieved_count"] = retrieved_count_list
    df["has_error"] = has_error_list

    # Cột RAGAS cần: answer (nếu chưa có) và ground_truth
    if "answer" not in df.columns:
        df["answer"] = ""          # Điền sau bằng LLM hoặc thủ công
    if "ground_truth" not in df.columns:
        df["ground_truth"] = ""    # Điền thủ công để RAGAS tính faithfulness/recall

    # Cột review thủ công
    df["is_valid_context"] = df["has_error"].apply(lambda x: not x)
    df["contexts_review"] = ""     # Ghi chú khi review thủ công

    # Sắp xếp cột theo thứ tự RAGAS-friendly
    priority_cols = ["question", "answer", "contexts", "ground_truth",
                     "sources", "contexts_preview", "retrieved_count",
                     "is_valid_context", "contexts_review", "has_error"]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]

    # ── Lưu file ───────────────────────────────────────────────────────────
    if output_excel is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        Path("data/evaluation").mkdir(parents=True, exist_ok=True)
        output_excel = f"data/evaluation/test_dataset_hybrid_{timestamp}.xlsx"

    df.to_excel(output_excel, index=False)

    # ── Tổng kết ───────────────────────────────────────────────────────────
    success_count = sum(1 for e in has_error_list if not e)
    error_count = sum(has_error_list)
    print(f"\n{'='*60}")
    print(f"✅ HOÀN TẤT!")
    print(f"   Tổng câu hỏi : {total}")
    print(f"   Thành công   : {success_count}")
    print(f"   Lỗi          : {error_count}")
    print(f"   Output file  : {output_excel}")
    print(f"{'='*60}")
    print("\n📋 Bước tiếp theo:")
    print("   1. Mở file Excel → review cột 'contexts_preview'")
    print("   2. Điền 'ground_truth' cho từng câu hỏi")
    print("   3. Điền 'answer' (chạy LLM) hoặc dùng generate_answers.py")
    print("   4. Đánh dấu 'is_valid_context' = False nếu context sai")
    print("   5. Chạy RAGAS evaluation với cột 'contexts' (JSON list)")

    return df


def load_ragas_dataset(excel_path: str):
    """
    Helper: Load file Excel đã tạo → chuyển về format dict cho RAGAS.

    Usage:
        from datasets import Dataset
        data = load_ragas_dataset("data/evaluation/test_dataset_hybrid_xxx.xlsx")
        ragas_dataset = Dataset.from_dict(data)
    """
    df = pd.read_excel(excel_path)

    # Filter valid rows
    if "is_valid_context" in df.columns:
        df = df[df["is_valid_context"] == True]

    # Parse JSON contexts
    df["contexts"] = df["contexts"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else []
    )

    # RAGAS cần dict với các key này
    ragas_data = {
        "question": df["question"].tolist(),
        "answer": df["answer"].fillna("").tolist(),
        "contexts": df["contexts"].tolist(),
        "ground_truth": df["ground_truth"].fillna("").tolist(),
    }

    print(f"Loaded {len(df)} valid samples cho RAGAS evaluation")
    return ragas_data


if __name__ == "__main__":
    generate_evaluation_dataset_with_hybrid()