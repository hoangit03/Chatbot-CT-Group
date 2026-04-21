from scripts.hybrid_retriever import HybridRetriever
import pandas as pd

def evaluate_retrieval(df: pd.DataFrame, retriever: HybridRetriever) -> dict:
    """
    df cần có cột: question, ground_truth_sources (file_name đúng)
    """
    hit_rates, precisions, recalls = [], [], []

    for _, row in df.iterrows():
        question = row["question"]
        expected_source = row["sources"]  # file_name chứa đáp án

        docs = retriever.retrieve(question, top_k=8)
        retrieved_sources = [d.metadata.get("file_name", "") for d in docs]

        # Hit Rate@K — có tìm được đúng source không?
        hit = any(expected_source in s for s in retrieved_sources)
        hit_rates.append(hit)

        # Precision@K — bao nhiêu % docs retrieve đúng?
        relevant = sum(1 for s in retrieved_sources if expected_source in s)
        precisions.append(relevant / len(retrieved_sources))

        # Recall@K — tìm được bao nhiêu docs đúng trong tổng số đúng?
        recalls.append(min(relevant, 1))  # simplified

    return {
        "hit_rate@8":   round(sum(hit_rates) / len(hit_rates), 4),
        "precision@8":  round(sum(precisions) / len(precisions), 4),
        "recall@8":     round(sum(recalls) / len(recalls), 4),
    }


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.services.embedder import Embedder

def evaluate_by_embedding_similarity(df: pd.DataFrame) -> dict:
    embedder = Embedder().get_embedding_model()

    scores = []
    for _, row in df.iterrows():
        if not row.get("answer") or not row.get("ground_truth"):
            continue

        emb_answer = embedder.embed_query(row["answer"])
        emb_gt     = embedder.embed_query(row["ground_truth"])

        score = cosine_similarity([emb_answer], [emb_gt])[0][0]
        scores.append(score)

    return {
        "avg_semantic_similarity": round(float(np.mean(scores)), 4),
        "min_similarity":          round(float(np.min(scores)), 4),
        "pct_above_0.8":           round(sum(s > 0.8 for s in scores) / len(scores), 4)
    }

from langchain_ollama import ChatOllama
import json

JUDGE_PROMPT = """Đánh giá câu trả lời RAG theo 3 tiêu chí. Trả về JSON.

Câu hỏi: {question}
Context: {context}
Câu trả lời: {answer}
Ground truth: {ground_truth}

Trả về ĐÚNG FORMAT sau, không thêm gì khác:
{{
  "faithfulness": <0.0-1.0, answer có bám context không>,
  "relevancy": <0.0-1.0, answer có đúng trọng tâm không>,
  "correctness": <0.0-1.0, answer có khớp ground truth không>,
  "reason": "<1 câu giải thích>"
}}"""

def llm_judge_evaluate(df: pd.DataFrame, model: str = "qwen3:1.7b") -> pd.DataFrame:
    llm = ChatOllama(model=model, temperature=0, format="json")
    results = []

    for _, row in df.iterrows():
        context_text = " ".join(row["contexts"][:2]) if row["contexts"] else ""
        prompt = JUDGE_PROMPT.format(
            question=row["question"],
            context=context_text[:1000],   # giới hạn để model không bị overwhelm
            answer=row.get("answer", ""),
            ground_truth=row.get("ground_truth", "")
        )
        try:
            response = llm.invoke(prompt)
            scores = json.loads(response.content)
        except Exception as e:
            scores = {"faithfulness": None, "relevancy": None,
                      "correctness": None, "reason": str(e)}

        results.append(scores)

    return pd.DataFrame(results)

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import datetime

def evaluate_lexical(df: pd.DataFrame) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rouge1_scores, rougeL_scores, bleu_scores = [], [], []

    for _, row in df.iterrows():
        answer = str(row.get("answer", ""))
        gt     = str(row.get("ground_truth", ""))
        if not answer or not gt:
            continue

        # ROUGE
        scores = scorer.score(gt, answer)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

        # BLEU
        ref    = [gt.split()]
        hyp    = answer.split()
        smooth = SmoothingFunction().method1
        bleu   = sentence_bleu(ref, hyp, smoothing_function=smooth)
        bleu_scores.append(bleu)

    return {
        "rouge1":  round(float(np.mean(rouge1_scores)), 4),
        "rougeL":  round(float(np.mean(rougeL_scores)), 4),
        "bleu":    round(float(np.mean(bleu_scores)), 4),
    }

def run_full_evaluation(excel_path: str):
    df = pd.read_excel(excel_path)
    df = df[df["is_valid_context"] == True]
    df["contexts"] = df["contexts"].apply(json.loads)

    df_temp = pd.read_csv('evaluation_report_20260420_112707.csv')

    df["answer"] = df_temp['response']
    print("=" * 60)

    # Tier 1: Không cần LLM — chạy ngay
    print("📏 [1/3] Lexical metrics...")
    lexical = evaluate_lexical(df)
    print(lexical)

    # Tier 2: Chỉ cần embeddings — nhanh
    print("🔢 [2/3] Embedding similarity...")
    semantic = evaluate_by_embedding_similarity(df)
    print(semantic)

    # Tier 3: LLM judge tự viết — kiểm soát được
    print("🤖 [3/3] LLM-as-Judge...")
    judge_df = llm_judge_evaluate(df)
    print(judge_df.mean(numeric_only=True).round(4))

    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {**lexical, **semantic,
               **judge_df.mean(numeric_only=True).round(4).to_dict()}

    with open(f"data/evaluation/summary_{timestamp}.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved: summary_{timestamp}.json")

if __name__ == "__main__":
    run_full_evaluation("data/evaluation/test_dataset_hybrid_20260420_0832.xlsx")
