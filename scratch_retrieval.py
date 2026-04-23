import chromadb

c = chromadb.HttpClient(host='localhost', port=8002)
col = c.get_collection('hr_policies')
res = col.query(query_texts=['Nắm rõ các OC, OPC, JD công việc của Phòng Ban mình và các Phòng Ban khác là nắm rỏ điều gì bạn ?'], n_results=15)

for i, (meta, doc) in enumerate(zip(res['metadatas'][0], res['documents'][0])):
    print(f"{i}: [{meta['source_file']}] - {doc[:50]}...")
