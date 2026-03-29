import os
from typing import Literal

import dotenv
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .cachembedding import CacheEmbedding

dotenv.load_dotenv()


class HybridTextSplitter:
    def __init__(
        self,
        cache_path,
        chunk_size=500,
        chunk_overlap=50,
        buffer_size=4,
        threshold_type: Literal[
            "percentile", "standard_deviation", "interquartile", "gradient"
        ] = "percentile",
        threshold_amount=95.0,
        similarity_threshold=0.97,
        enable_filter=True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = CacheEmbedding(cache_path)
        self.buffer_size = buffer_size
        self.threshold_type = threshold_type
        self.threshold_amount = threshold_amount
        self.similarity_threshold = similarity_threshold
        self.enable_filter = enable_filter

        self.rough_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,
            chunk_overlap=self.chunk_overlap * 2,
            separators=["\n\n", "\n", "。", "！", "？"],
        )
        self.lens_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=os.getenv("MODEL_NAME"),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，"],
        )
        if self.enable_filter:
            self.filter = EmbeddingsRedundantFilter(
                embeddings=self.embedding_model,
                similarity_threshold=self.similarity_threshold,
            )

    def split(self, documents: list[Document]) -> list[Document]:
        print("Step 1️⃣ 粗切分 ...")
        results = self.rough_splitter.split_documents(documents)
        print(f"  → 粗切分结果: {len(results)} 段")

        print("Step 2️⃣ 长度控制切分 ...")
        results = self.lens_splitter.split_documents(results)
        print(f"  → 长度控制后: {len(results)} 段")

        if self.enable_filter:
            print("Step 3️⃣ 冗余过滤 (EmbeddingsRedundantFilter) ...")
            results = self.filter.transform_documents(results)
            print(f"  → 去重后: {len(results)} 段")

        print("✅ 切分完成")
        return results
