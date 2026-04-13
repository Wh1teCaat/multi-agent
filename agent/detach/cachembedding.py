import hashlib
import json
import os
from functools import lru_cache

import dotenv
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()


class CacheEmbedding(Embeddings):
    """包装原始 embedding 模型，实现缓存 + 批量嵌入"""

    def __init__(
        self,
        cache_path,
        batch_size=128,
    ):
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("HF_MODEL_NAME"),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": self.batch_size, "normalize_embeddings": True},
        )

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = f.read().strip()
                    if not data:
                        return {}
                    return json.loads(data)
            except json.JSONDecodeError:
                print(f"⚠️ 缓存文件损坏 ({self.cache_path})，已重置为空缓存。")
                return {}
            except Exception as e:
                print(f"⚠️ 加载缓存时出现错误: {e}")
                return {}
        return {}

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @lru_cache(maxsize=None)
    def embed_query(self, text: str) -> list[float]:
        _hash = self._text_hash(text)
        if _hash in self.cache:
            return self.cache[_hash]
        vec = self.embeddings.embed_query(text)
        self.cache[_hash] = vec
        self._save_cache()
        return vec

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        results, to_compute = [], []
        for text in texts:
            _hash = self._text_hash(text)
            if _hash in self.cache:
                results.append(self.cache[_hash])
            else:
                to_compute.append(text)

        if to_compute:
            vectors = self.embeddings.embed_documents(to_compute)
            for t, v in zip(to_compute, vectors):
                _hash = self._text_hash(t)
                self.cache[_hash] = v
                results.append(v)
            self._save_cache()
        return results

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        for batch in batches:
            results.extend(self._embed_batch(batch))
        return results
