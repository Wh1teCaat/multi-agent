from pathlib import Path

import yaml
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# project root: /home/y2/multi-agent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

from tools.base_tool import BaseToolWrapper
from detach.retriever import RAG


config_path = PROJECT_ROOT / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_path = PROJECT_ROOT / config["loader"]["data_path"]
cache_path = PROJECT_ROOT / config["embedding"]["cache_path"]
db_path = PROJECT_ROOT / config["retriever"]["db_path"]


class RagTool(BaseToolWrapper):
    DEFAULT_NAME = "RagTool"
    DEFAULT_DESC = """
    当用户的问题需要查找知识库中具体信息（例如论文、技术内容、事实说明等）时使用此工具。
    仅当问题属于“知识问答、专业内容、事实查询”时调用；
    对于闲聊、反问、总结、情绪、历史对话类问题，请直接回答，不要调用本工具。
    """

    def __init__(self, data_path, db_path, cache_path):
        super().__init__()
        self.data_path = data_path
        self.db_path = db_path
        self.cache_path = cache_path

    def build(self):
        rag = RAG(self.data_path, self.db_path, self.cache_path)
        retriever = rag.get_retriever()

        class ArgSchema(BaseModel):
            query: str = Field(description="用户输入内容")

        async def _rag_func_async(query: str):
            return await retriever.ainvoke(query)

        def _rag_func_sync(query: str):
            return retriever.invoke(query)

        return StructuredTool.from_function(
            func=_rag_func_sync,
            coroutine=_rag_func_async,
            name=self.name,
            description=self.description,
            arg_schema=ArgSchema,
            return_direct=False,
        )


rag_retriever = RagTool(data_path, db_path, cache_path).build()
