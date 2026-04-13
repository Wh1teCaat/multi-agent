import asyncio
import os
import uuid
from typing import TypedDict, List, Annotated, Literal, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel, Field

from detach.tools.rag_tool import rag_retriever


class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: Optional[List[Document]]
    question: str
    retry_count: int
    grade: Optional[str]


async def retrieve(state: RAGState):
    question = state["question"]
    docs = await rag_retriever.ainvoke(question)
    return {"documents": docs}


class Grade(BaseModel):
    grade: Literal["yes", "no"] = Field(description="只回答 'yes' or 'no'")


llm = ChatOpenAI(model=os.getenv("MODEL_NAME"))
structured_llm = llm.with_structured_output(Grade)


async def grade_documents(state: RAGState):
    question = state["question"]
    documents = state.get("documents") or []
    template = PromptTemplate.from_template(
        """
    你是一个评审员。
    这是用户的问题：{question}
    这是检索的文档：{document}
    请判断这个文档真回答了问题吗？
    """
    )
    tasks = []
    for document in documents:
        prompt = template.format(question=question, document=document.page_content)
        task = structured_llm.ainvoke(prompt)
        tasks.append(task)

    results = await asyncio.gather(*tasks) if tasks else []
    grades = [res.grade for res in results]

    reduced_docs = []
    for doc, grade in zip(documents, grades):
        if grade == "yes":
            reduced_docs.append(doc)

    if not reduced_docs:
        return {
            "documents": [],
            "grade": "no",
        }
    return {
        "documents": reduced_docs,
        "grade": "yes",
    }


async def generate(state: RAGState):
    question = state["question"]
    documents = state.get("documents") or []

    if not documents:
        final_answer = "抱歉，经过多次检索，我依然没有在知识库中找到与该问题相关的信息。建议您尝试更换关键词或查阅其他来源。"
        return {"messages": [AIMessage(content=final_answer)]}

    docs = "\n\n".join(doc.page_content for doc in documents)
    prompt = f"""
    这是用户的提问：{question}
    这是 RAG 检索到的相关信息：{docs}
    请你根据这些信息回答用户的问题。
    """
    result = await llm.ainvoke(prompt)
    return {"messages": [result]}


async def rewrite(state: RAGState):
    question = state["question"]
    current_attempt = state.get("retry_count", 0)
    prompt = f"""
    用户的问题是：{question}
    初次检索没有发现相关信息。
    请分析问题意图，输出一个优化后的、更适合搜索引擎的关键词。
    只输出关键词，不要包含解释。
    """
    result = await llm.ainvoke(prompt)
    print(f"🔄 改写问题: {question} -> {result.content} (第 {current_attempt + 1} 次尝试)")
    return {
        "question": result.content,
        "retry_count": current_attempt + 1,
    }


graph = StateGraph(RAGState)
graph.add_node("rag", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("rewrite", rewrite)
graph.add_node("generate", generate)
graph.set_entry_point("rag")
graph.add_edge("rag", "grade")


def grade_continue(state: RAGState):
    grade = state["grade"]
    retry_count = state["retry_count"]
    if grade == "yes":
        return "generate"
    if retry_count < 3:
        return "rewrite"
    return "generate"


graph.add_conditional_edges("grade", grade_continue)
graph.add_edge("rewrite", "rag")
graph.add_edge("generate", "__end__")
app = graph.compile()


@tool
async def call_rag_expert(task: str) -> str:
    """
    【内部知识库专家】

    适用场景：
    1. 查询专业领域知识（如地质、法律、公司规章等本地文档）。
    2. 查询历史档案或已有的固定事实。

    ❌ 严禁用于：
    1. 查询实时信息（如今天的天气、现在的股价）。
    2. 查询未来的预测（如2025年的事情）。
    3. 闲聊。
    """
    inputs = {
        "messages": [HumanMessage(content=task)],
        "question": task,
        "retry_count": 0,
        "documents": [],
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = await app.ainvoke(inputs, config)

    final_msg = result["messages"][-1]
    return final_msg.content
