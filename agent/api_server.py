import asyncio

import os
os.environ["USER_AGENT"] = "my-agent-server/1.0"

import uvicorn
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from typing import List, Any
from typing_extensions import TypedDict

from agent import Agent, Receipt

class ChatRequest(BaseModel):
    query: str = Field(description="用户提问的内容")
    thread_id: str = Field(description="会话ID，用于区分不同用户", default="default_thread")

# 生命周期管理（Lifespan）
# FastAPI 的核心特性：在服务器启动前建立连接，关闭后释放连接
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    print("正在初始化 Agent 及数据库连接...")
    agent_instance = await Agent.create()
    print("✅ 系统就绪，数据库已连接。")

    yield

    print("正在关闭数据库连接...")
    if agent_instance:
        await agent_instance.aclose()
    print("🛑 系统已关闭。")

app = FastAPI(
    title="AI Agent Server",
    version="1.0",
    description="基于 LangGraph + PostgreSQL 的异步智能体服务",
    lifespan=lifespan
)

@app.get("/")
async def redirect_root():
    return RedirectResponse("/docs")

# 接口 A: 简单直观的自定义接口 (供前端 App/小程序调用)
# URL: POST http://localhost:8000/chat
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = await agent_instance.ainvoke(
        query=request.query,
        thread_id=request.thread_id,
    )

    if type(response) == Receipt:
        return {
            "reason": response.reason,
            "answer": response.answer,
            "source": response.source,
        }
    return {"answer": response}

class AgentInput(TypedDict):
    query: str
    thread_id: str

# 接口 B: LangServe 标准接口 (供调试、LangSmith 或高级流式前端调用)
# URL: http://localhost:8000/agent/playground
# 注意：这里通过一个 wrapper 函数来暴露 Agent 的能力
async def langserve_wrapper(inputs: AgentInput):
    # LangServe 传进来的 inputs 通常是 {"messages": [...]}
    # 我们提取最后一条消息作为 query
    query = inputs["query"]
    thread_id = inputs.get("thread_id", "default_thread")
    return await agent_instance.ainvoke(
        query=query,
        thread_id=thread_id,
)

langserve_runnable = RunnableLambda(langserve_wrapper)

add_routes(
    app,
    langserve_runnable,
    path="/agent",
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
