from app.services.retrieval import RetrievalService
from app.services.embedder import Embedder

ret = RetrievalService()
# Vô hiệu hóa reranker để xem thuần túy VectorDB trả về gì
ret.reranker = None 

res = ret.retrieve("Nắm rõ các OC, OPC, JD công việc của Phòng Ban mình và các Phòng Ban khác là nắm rỏ điều gì bạn ?")
for i, d in enumerate(res.documents):
    print(f"[{i}] {d.metadata.get('source_file')} - Score: {d.metadata.get('similarity_score')} - {d.page_content[:50]}")
