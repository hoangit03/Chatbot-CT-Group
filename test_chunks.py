import json, glob, os

# Find the chunks file
chunks_dir = r"D:\CTGroup\CT_Knowledge\XuLy_Data_Rag_CTG\shared_data\data_output\result_chunks"
target = "B.NLC"

for f in os.listdir(chunks_dir):
    if target in f:
        path = os.path.join(chunks_dir, f)
        print(f"File: {f}")
        chunks = json.load(open(path, encoding='utf-8'))
        print(f"Total chunks: {len(chunks)}\n")
        
        for i, c in enumerate(chunks):
            content = c["content"]
            # Show full content for chunks near "nghi viec huong luong"
            if "nghỉ việc hưởng" in content or "nghỉ hưởng" in content or "Nghỉ phép năm" in content.replace("\\n", "\n"):
                print(f"=== CHUNK {i} ({len(content)} chars) [RELEVANT] ===")
                print(content[:600])
                print("...\n")
            else:
                print(f"--- Chunk {i} ({len(content)} chars): {content[:80].replace(chr(10), ' ')}...")
        break
