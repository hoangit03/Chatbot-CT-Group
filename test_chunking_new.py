import sys, os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pipeline.engines.chunking_engine import run_chunking_pipeline

# Test với file phúc lợi
test_file = "B.NLCĐ- - Quy dinh chinh sach phuc loi-tham khao.pdf.md"
result = run_chunking_pipeline(test_file)

if result:
    import json
    chunks_dir = os.path.join(os.getenv("SHARED_DATA_DIR", "shared_data"), "data_output", "result_chunks")
    # Find the output file
    for f in os.listdir(chunks_dir):
        if "B.NLC" in f:
            chunks = json.load(open(os.path.join(chunks_dir, f), encoding='utf-8'))
            print(f"\n{'='*60}")
            print(f"Total chunks: {len(chunks)}")
            print(f"{'='*60}")
            
            for i, c in enumerate(chunks):
                content = c["content"]
                if any(kw in content for kw in ["nghỉ việc hưởng", "nghỉ hưởng", "Nghỉ phép năm", "100% lương"]):
                    print(f"\n=== CHUNK {i} ({len(content)} chars) [RELEVANT] ===")
                    print(content[:500])
                    print("..." if len(content) > 500 else "")
                else:
                    print(f"--- Chunk {i} ({len(content)} chars): {content[:80].replace(chr(10), ' ')}...")
            break
