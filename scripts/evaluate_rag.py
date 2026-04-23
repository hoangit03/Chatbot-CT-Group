import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

from app.services.rag_service import RAGService


def load_test_dataset(file_path: str = "data/evaluation/test_dataset.json") -> List[Dict]:
    f = pd.read_excel(file_path)
    df = df[df["is_valid_context"] == True]  # lọc row lỗi
    df["contexts"] = df["contexts"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    return df.to_dict("records")


def run_evaluation(excel_path: str):
    print("BẮT ĐẦU ĐÁNH GIÁ RAG BẰNG RAGAS...\n")

    # 1. Load test dataset
    df = pd.read_excel(excel_path)
    # print(f"Đã load {len(df)} câu hỏi đánh giá\n")
    # answers = []


    # rag_service = RAGService()

    # for _, row in df.iterrows():
    #     question = row["question"]
    #     # Chỉ generate answer — KHÔNG retrieve lại
    #     result = rag_service.answer(query=question)
    #     answers.append(result["answer"])
    df_temp = pd.read_csv('evaluation_report_20260420_112707.csv')

    df["answer"] = df_temp['response']
    # 3. Tạo Dataset cho RAGAS
    dataset = Dataset.from_dict({
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": df["contexts"].apply(json.loads).tolist(),
        "ground_truth": df["ground_truth"].tolist(),
    })

    # 4. Chạy đánh giá
    print("\n🔬 Đang chạy RAGAS evaluation...")

    judge_llm = LangchainLLMWrapper(ChatOllama(model="qwen3:1.7b", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    )

    # mini = Dataset.from_dict({
    #     "question": ["Công ty thành lập năm nào?"],
    #     "answer": ["Công ty thành lập năm 2010."],
    #     "contexts": [["CT-Group thành lập năm 2010 tại TP.HCM."]],
    #     "ground_truth": ["Công ty thành lập năm 2010"]
    # })
    
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ],
        llm=judge_llm,
        embeddings=judge_embeddings ,
        run_config=RunConfig(
            max_retries=2,
            timeout=300,
            max_workers=1
        )
    )

    # 5. Hiển thị & lưu kết quả
    df = result.to_pandas()
    print("\n" + "="*80)
    print("KẾT QUẢ ĐÁNH GIÁ RAG")
    print("="*80)
    print(df.mean(numeric_only=True).round(4))

    # Lưu báo cáo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"evaluation_report_{timestamp}.csv"
    df.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"\nĐã lưu báo cáo chi tiết: {report_path}")
    print("Hoàn tất đánh giá RAG!")

if __name__ == "__main__":
    run_evaluation(r"D:\AnChamLam\CT-Group\Chatbot\Final-version\Chatbot-CT-Group\test_dataset_hybrid_20260420_0832.xlsx")