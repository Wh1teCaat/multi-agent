import hashlib
import os
from enum import Enum

from langchain_chroma import Chroma

from .cachembedding import CacheEmbedding
from .hybridtextsplitter import HybridTextSplitter
from .multiloader import MultiLoader


class RunMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class RAG:
    def __init__(self, data_path, db_path, cache_path, mode=RunMode.ONLINE):
        self.data_path = data_path
        self.db_path = db_path
        self.cache_path = cache_path
        self.loader = MultiLoader(self.data_path)
        self.splitter = HybridTextSplitter(self.cache_path)
        self.embedding = self.splitter.embedding_model
        self.mode = mode

    def _process_documents(self):
        docs = self.loader.load()
        print("文件加载完成")

        docs = self.splitter.split(docs)
        print("文档切分完成")
        return docs

    @staticmethod
    def make_md5(text: str):
        if not text:
            return ""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _build_db(self):
        docs = self._process_documents()
        db = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=self.db_path,
        )
        print("✅ 向量数据库构建完成")
        return db

    def _append_db(self, db):
        docs = self._process_documents()
        exist_docs = set(
            m.get("hash") for m in db.get(include=["metadatas"])["metadatas"] if m.get("hash")
        )
        docs = [d for d in docs if self.make_md5(d.page_content) not in exist_docs]

        if not docs:
            print("🟡 没有检测到新文档，数据库无需更新")
            return db

        db.add_documents(documents=docs)
        return db

    def get_retriever(self):
        if not os.path.exists(self.db_path) or not os.listdir(self.db_path):
            print("⚠️ 未检测到持久化文件，正在重新构建数据库...")
            if self.mode == RunMode.OFFLINE:
                db = self._build_db()
            else:
                raise RuntimeError("❌ 在线模式下无法构建新数据库，请先运行离线模式初始化")
        else:
            print("✅ 加载已有数据库...")
            db = Chroma(
                persist_directory=self.db_path,
                embedding_function=CacheEmbedding(self.cache_path),
            )

            if self.mode == RunMode.OFFLINE:
                db = self._append_db(db)

        retriever = db.as_retriever()
        return retriever
