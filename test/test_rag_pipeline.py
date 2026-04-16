from langchain_core.messages import HumanMessage, AIMessage
from app.services.rag_service import RAGService

service = RAGService()

history = [
]

result = service.answer(
    query="CHƯƠNG TRÌNH PHÚC LỢI",
    chat_history=history
)

print("Câu trả lời:", result["answer"])
print("Nguồn:", result["sources"])