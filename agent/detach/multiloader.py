import hashlib
import os

from datasets import load_dataset
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class MultiLoader(BaseLoader):
    """加载指定目录下的 huggingface 数据集和本地文件"""

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    @staticmethod
    def _convert_huggingface_path(dirname: str) -> str:
        return dirname.replace("___", "/").replace("---", "/")

    @staticmethod
    def _is_huggingface_path(filename: str) -> bool:
        return "___" in filename or "---" in filename or "/" in filename

    @staticmethod
    def make_md5(text: str):
        if not text:
            return ""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_file(self, filename: str, sample_num=100) -> list[Document]:
        if self._is_huggingface_path(filename):
            print(f"😀加载 HuggingFace 数据集：{filename}")
            try:
                dataset = load_dataset(
                    filename,
                    split="train",
                    cache_dir="./data/huggingface",
                )
                sample = dataset.shuffle().select(range(min(sample_num, len(dataset))))
            except Exception as e:
                raise RuntimeError(f"❌ 加载数据集失败: {e}")

            docs = [
                Document(
                    page_content=record.get("positive_doc")[0].get("content"),
                    metadata={
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "datatype": record.get("positive_doc")[0].get("datatype"),
                        "title": record.get("positive_doc")[0].get("title"),
                        "hash": self.make_md5(
                            record.get("positive_doc")[0].get("content")
                        ),
                    },
                )
                for record in sample
            ]
            return docs

        path = os.path.join(self.path, filename)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            sub_loader = TextLoader(path, encoding="utf-8")
        elif ext == ".pdf":
            sub_loader = PyPDFLoader(path)
        elif ext == ".csv":
            sub_loader = CSVLoader(path)
        elif ext == ".json":
            sub_loader = JSONLoader(
                file_path=path,
                jq_schema="""
                .[] | {
                    question : .question,
                    answer : .answer,
                    content : .context
                }
                """,
                metadata_func=lambda record, metadata: {
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "source": metadata.get("source"),
                },
                content_key="content",
            )
        elif ext == ".html":
            sub_loader = UnstructuredHTMLLoader(path, mode="elements", strategy="fast")
        elif ext == ".md":
            sub_loader = UnstructuredMarkdownLoader(path, strategy="fast")
        else:
            return [
                Document(
                    page_content="",
                    metadata={"source": path, "error": "unsupported file type"},
                )
            ]
        try:
            return sub_loader.load()
        except Exception as e:
            return [Document(page_content="", metadata={"source": path, "error": str(e)})]

    def load(self):
        items = os.listdir(self.path)
        docs = []
        for item in items:
            if item == "huggingface":
                dirs = os.listdir(os.path.join(self.path, item))
                for dirname in dirs:
                    path = self._convert_huggingface_path(dirname)
                    docs.extend(self._load_file(path))
            else:
                docs.extend(self._load_file(item))
        return docs
