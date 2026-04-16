import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.utils.multi_document_loader import MultiDocumentLoader

load_dotenv()

def main():
    print("BẮT ĐẦU INGESTION TỪ data/raw (multi-format + recursive)\n")

    # 1. Load tất cả documents
    print("1. Đang load documents từ data/raw...")
    loader = MultiDocumentLoader(raw_dir="data/raw")
    docs = loader.load_all()

    if not docs:
        print("Không tìm thấy file nào hợp lệ!")
        return

    # 2. Chunking
    print("2. Đang chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"   → Tạo {len(splits)} chunks")

    # 3. Embedding + Vector Store
    print("3. Đang embedding (multilingual-e5-large) và lưu vào Chroma...")
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    )

    persist_dir = os.getenv("VECTOR_DB_DIR", "./vectorstore/db")

    # Xóa DB cũ nếu muốn reset 
    if Path(persist_dir).exists():
        import shutil
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="internal_knowledge"
    )
    vectorstore.persist()

    print(f"INGESTION HOÀN TẤT!")
    print(f"   Vector Store: {persist_dir}")
    print(f"   Số chunks: {len(splits)}")
    print(f"   Sẵn sàng dùng cho RAG!")

if __name__ == "__main__":
    main()