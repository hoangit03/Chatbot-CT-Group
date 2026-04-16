import sys
from pathlib import Path
from typing import List

from app.services.ingestion_service import IngestionService


def test_full_ingestion():
    """Test full ingestion (toàn bộ data/raw)"""
    print("=" * 60)
    print("TEST 1: FULL INGESTION (run_full)")
    print("=" * 60)

    try:
        service = IngestionService()
        service.run_full(replace=True)  
        print("\nFULL INGESTION TEST HOÀN TẤT THÀNH CÔNG!")
    except Exception as e:
        print(f"Lỗi khi full ingestion: {type(e).__name__} - {e}")


def test_single_file():
    """Test ingest chỉ 1 file duy nhất"""
    print("\n" + "=" * 60)
    print("TEST 3: INGEST SINGLE FILE")
    print("=" * 60)

    single_file = "data/raw/test_file.pdf"   # ← thay bằng file thật của bạn

    try:
        service = IngestionService()
        service.ingest_documents([single_file])
        print("\nSINGLE FILE INGESTION TEST HOÀN TẤT THÀNH CÔNG!")
    except Exception as e:
        print(f"Lỗi khi ingest single file: {type(e).__name__} - {e}")


if __name__ == "__main__":
    print("CHẠY TEST INGESTION SERVICE\n")
    
    test_full_ingestion()           # Test full
    # test_single_file()            # Test 1 file

    print("\n" + "=" * 60)
    print("HOÀN TẤT TẤT CẢ CÁC TEST")