import math
import os
import uuid
from datetime import datetime
from typing import TypedDict, List, Annotated

import dotenv
os.environ.setdefault("USER_AGENT", "multi-agent/1.0")
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import add_messages, StateGraph

dotenv.load_dotenv()

@tool
async def get_current_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    获取当前时间。
    当用户问“今天”、“明天”、“周末”等与时间相关的问题时，必须先调用此工具获取基准时间。
    """
    return datetime.now().strftime(format)  # 按 format 字符串格式化时间

@tool
async def calculator(expression: str) -> str:
    """
    一个计算器工具。
    适用于计算具体的数学表达式，如 '234 * 45' 或 'sqrt(100)'。
    注意：仅支持简单的数学运算，不要输入代码。
    """
    safe_dict = {
        "math": math,
        "sqrt": math.sqrt,
        "pow": math.pow,
        "pi": math.pi,
        "sin": math.sin,
        "cos": math.cos,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "__builtins__": {}
    }
    try:
        result = eval(expression, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
async def scrape_webpage(url: str) -> str:
    """
    抓取并读取指定 URL 网页的详细文本内容。
    当你通过搜索获得了链接，但需要了解链接里的具体细节时，调用此工具。
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        content = "\n\n".join(doc.page_content for doc in docs)
        content = "\n".join(line.strip() for line in content.split("\n") if line.strip())
        return content[:3000]
    except Exception as e:
        return f"Error: {e}"

tavily = TavilySearch(max_results=3)
tools = [get_current_time, calculator, scrape_webpage, tavily]
tools_by_name = {tool.name: tool for tool in tools}
llm = ChatOpenAI(model=os.getenv("MODEL_NAME"))
llm_with_tools = llm.bind_tools(tools)


class SearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# tools node
async def tools_node(state: SearchState):
    last_msg = state["messages"][-1]

    if not last_msg.tool_calls:
        return {}

    tool_msgs = []
    for tool_call in last_msg.tool_calls:
        name = tool_call["name"]

        if name not in tools_by_name:
            output = f"Error: Tool {name} not found"
        else:
            tool_func = tools_by_name[name]
            tool_args = tool_call["args"]
            try:
                output = await tool_func.ainvoke(tool_args)
            except Exception as e:
                output = f"Error: {e}"

        tool_msgs.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": tool_msgs}

# agent node
async def agent_node(state: SearchState):
    messages = state["messages"]
    result = await llm_with_tools.ainvoke(messages)
    return {"messages": [result]}

graph = StateGraph(SearchState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.set_entry_point("agent")
graph.add_edge("tools", "agent")

def agent_continue(state: SearchState):
    last_msg = state["messages"][-1]

    if last_msg.tool_calls:
        return "tools"
    else:
        return "__end__"

graph.add_conditional_edges("agent", agent_continue)
app = graph.compile()

@tool
async def call_search_expert(task: str) -> str:
    """
    【互联网搜索专家】

    适用场景：
    1. 查询实时新闻、天气预报、股票价格。
    2. 查询由于时间推移可能变化的信息（如汇率、总统是谁）。
    3. 需要进行数学计算的问题。
    4. 查询 RAG 知识库中没有的广泛互联网信息。

    示例：
    - "明天北京天气怎么样？" -> 调用此工具。
    - "2024年奥运会金牌榜" -> 调用此工具。
    """
    inputs = {"messages": [HumanMessage(content=task)]}
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = await app.ainvoke(inputs, config)

    return result["messages"][-1].content
