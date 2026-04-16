from app.utils.multi_document_loader import MultiDocumentLoader
from app.utils.preprocessor import TextPreprocessor
from app.services.document_splitter import RecursiveSplitter
import os
from dotenv import load_dotenv

load_dotenv()

print("1. Đang load documents từ data/raw...")
raw_dir = os.getenv("DIR_DATA_RAW", "data/raw")
loader = MultiDocumentLoader(raw_dir=raw_dir)
docs = loader.load_all()
preprocessor = TextPreprocessor()
splitter = RecursiveSplitter()
docs = preprocessor.process_documents(docs)
splits = splitter.split(docs)

if not docs:
    print("Không tìm thấy file nào hợp lệ!")

print(len(docs))
for i in range(0, 5):
    print(docs[i])
    print("#"*30)
    print(splits[i])
