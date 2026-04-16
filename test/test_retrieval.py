from app.services.retrieval import RetrievalService


def test_retrieval():
    print("TEST RETRIEVAL + RERANKER + GPU\n")
    
    service = RetrievalService()

    queries = [
        "CHƯƠNG TRÌNH PHÚC LỢI",
        "Môi trường làm việc ở công ty",
        "Thủ tục xin nghỉ thai sản"
    ]

    for i, q in enumerate(queries, 1):
        print(f"\nQuery {i}: {q}")
        result = service.retrieve(query=q)
        
        for j, doc in enumerate(result.documents, 1):
            score = doc.metadata.get("rerank_score") or doc.metadata.get("similarity_score", 0)
            print(f"   {j}. [{score:.4f}] {doc.metadata.get('file_name', 'Unknown')}")
            print(f"       {doc.page_content[:120]}...\n")

    print("TEST RETRIEVAL HOÀN TẤT")


if __name__ == "__main__":
    test_retrieval()