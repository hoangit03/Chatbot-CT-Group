import os
import json
import redis
import numpy as np
from typing import Optional
from numpy.linalg import norm
from dotenv import load_dotenv

load_dotenv()

class SemanticCache:
    """
    Dịch vụ Semantic Cache sử dụng Redis.
    Lưu trữ Vector Embedding của các câu hỏi cũ và câu trả lời tương ứng.
    Nếu câu hỏi mới có độ tương đồng > 0.95 với câu hỏi cũ -> Trả về thẳng Cache.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Mặc định lấy từ biến môi trường
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.95))
        self.enabled = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
        self.client = None
        
        if self.enabled:
            try:
                self.client = redis.Redis(
                    host=redis_host, 
                    port=redis_port, 
                    decode_responses=False
                )
                self.client.ping()
                print(f"[SemanticCache] Redis kết nối thành công tại {redis_host}:{redis_port}.")
            except Exception as e:
                print(f"[SemanticCache] Redis kết nối thất bại: {e}. Vô hiệu hóa Cache.")
                self.enabled = False

    def _cosine_similarity(self, vec1, vec2):
        if not len(vec1) or not len(vec2):
            return 0.0
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_product = norm(vec1) * norm(vec2)
        if norm_product == 0:
            return 0.0
        return dot_product / norm_product

    def search_cache(self, query_embedding: list) -> Optional[str]:
        """Tìm kiếm câu hỏi tương tự trong Cache."""
        if not self.enabled or not self.client:
            return None
            
        try:
            keys = self.client.keys("cache:semantic:*")
            best_match = None
            highest_score = 0.0
            
            vec1 = np.array(query_embedding)
            
            for key in keys:
                data_bytes = self.client.get(key)
                if data_bytes:
                    data = json.loads(data_bytes.decode('utf-8'))
                    cached_vec = np.array(data["embedding"])
                    score = self._cosine_similarity(vec1, cached_vec)
                    
                    if score > highest_score:
                        highest_score = score
                        best_match = data["answer"]
                        
            if highest_score >= self.threshold:
                print(f"[SemanticCache] 🟢 HIT! Tương đồng: {highest_score:.3f}")
                return best_match
            
            print(f"[SemanticCache] 🔴 MISS! Độ tương đồng cao nhất: {highest_score:.3f}")
            return None
        except Exception as e:
            print(f"[SemanticCache] Lỗi tìm kiếm: {e}")
            return None

    def add_to_cache(self, query_embedding: list, answer: str):
        """Lưu kết quả RAG vào Cache để dùng cho lần sau."""
        if not self.enabled or not self.client:
            return
            
        try:
            import uuid
            key = f"cache:semantic:{uuid.uuid4()}"
            data = {
                "embedding": query_embedding,
                "answer": answer
            }
            # Cache lưu 24h
            self.client.setex(key, 86400, json.dumps(data).encode('utf-8'))
            print("[SemanticCache] Đã lưu vào Cache thành công.")
        except Exception as e:
            print(f"[SemanticCache] Lỗi lưu Cache: {e}")

    def flush_cache(self):
        """Xóa toàn bộ Cache khi cập nhật tài liệu (Wipe / Force Update)."""
        if self.client:
            try:
                keys = self.client.keys("cache:semantic:*")
                if keys:
                    self.client.delete(*keys)
                print("[SemanticCache] 🧹 Đã Flush toàn bộ Semantic Cache.")
            except Exception as e:
                print(f"[SemanticCache] Lỗi Flush Cache: {e}")
