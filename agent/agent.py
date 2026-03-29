import ast
import os
import re
from typing import Any, TypedDict, Annotated, Optional

import dotenv
import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from tool_registry import build_default_tool_specs, build_tool_maps
from tools.task_tools import interactive_yes_no

dotenv.load_dotenv()


RECALL_PATTERNS = [
    r"刚刚",
    r"上一次|上个",
    r"之前",
    r"回顾",
    r"总结上文|复述上文|回忆",
    r"task\s*\d+.*(结论|结果|说了什么)",
    r"我刚才.*(说|问)",
    r"结论是什么",
    r"结果是什么",
]


def _is_recall_query(text: str) -> bool:
    if not text:
        return False
    normalized = text.strip().lower()
    return any(re.search(p, normalized) for p in RECALL_PATTERNS)


class Receipt(BaseModel):
    """结构化输出"""
    reason: str = Field(
        default=None,
        description="""
        【思维链分析】
        1. 用户最新一句话的意图是什么？（是延续上文，还是开启新任务？）
        2. 如果需要回忆，请提取历史消息中的关键信息。
        3. 解释为什么选择调用（或不调用）某个工具。
        """
    )
    answer: str = Field(
        description="""
        针对用户问题的最终回答内容。
        【重要警告】：
        - 如果用户要求写作文、写代码、写长文，此字段**必须包含完整的生成内容（全文）**。
        - **严禁**只输出一句“已生成作文”或“见下文”之类的摘要。
        - 必须是用户想看的那个结果本身。
        """
    )
    source: list[str] = Field(description="回答中引用的具体文档名称或页码列表。如果没用到文档，请留空。")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: Optional[str]
    structured_answer: Optional[Receipt]


class Agent:
    def __init__(self, runnable, pool):
        self.runnable = runnable
        self.pool = pool
        self.thread_answer_cache: dict[str, list[str]] = {}

    @classmethod
    async def create(cls, max_tokens=5000):
        max_tokens = max_tokens
        tool_specs = build_default_tool_specs()
        tools, tools_by_name, specs_by_name = build_tool_maps(tool_specs)
        llm = ChatOpenAI(model=os.getenv("MODEL_NAME"))
        llm_with_tools = llm.bind_tools(tools)
        llm_structured = llm.with_structured_output(Receipt)

        def _last_user_text(messages: list[BaseMessage]) -> str:    # 取出消息列表最后的用户消息文本
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
                    return msg.content
            return ""

        def _sanitize_tool_call_sequence(messages: list[BaseMessage]) -> list[BaseMessage]:
            """修复历史中不完整的 tool_call -> tool_message 序列。"""
            fixed: list[BaseMessage] = []
            i = 0
            total = len(messages)

            while i < total:
                msg = messages[i]
                fixed.append(msg)

                calls = msg.tool_calls if isinstance(msg, AIMessage) else None
                if not calls:
                    i += 1
                    continue

                required_ids = [c.get("id") for c in calls if c.get("id")]
                seen_ids: set[str] = set()

                j = i + 1
                while j < total and isinstance(messages[j], ToolMessage):
                    tool_msg = messages[j]
                    fixed.append(tool_msg)
                    if tool_msg.tool_call_id:
                        seen_ids.add(tool_msg.tool_call_id)
                    j += 1

                for call_id in required_ids:
                    if call_id not in seen_ids:
                        fixed.append(
                            ToolMessage(
                                content="Error: 历史会话缺失该 tool_call 的执行回执，已自动补齐占位回执。",
                                tool_call_id=call_id,
                            )
                        )

                i = j

            return fixed

        async def _human_confirm_tool_call(tool_name: str, args: dict[str, Any], confirm_key: str) -> bool | None:
            """人在回路确认：返回 True/False；若环境不支持交互返回 None。"""
            preview = {k: v for k, v in args.items() if k != confirm_key}
            prompt = (
                f"[agent] 工具 {tool_name} 需要人工确认。\n"
                f"[agent] 参数预览: {preview}\n"
                f"[agent] 是否继续并设置 {confirm_key}=true ? (y/n): "
            )
            return await interactive_yes_no(prompt)

        async def _structured_node(state: AgentState):
            messages = state["messages"]
            summary = state.get("summary", "")
            messages = _sanitize_tool_call_sequence(messages)

            if summary:
                prompt_msg = [
                    SystemMessage(content=f"上下文摘要：{summary}"),
                    SystemMessage(content="若上文存在 run_command 的工具回执，必须严格基于回执给出结论；禁止让用户再次手工去本地执行同一命令。"),
                ] + messages
            else:
                prompt_msg = [
                    SystemMessage(content="若上文存在 run_command 的工具回执，必须严格基于回执给出结论；禁止让用户再次手工去本地执行同一命令。"),
                ] + messages

            receipt = await llm_structured.ainvoke(prompt_msg)
            return {"structured_answer": receipt}

        async def _summary_node(state: AgentState):
            """摘要逻辑节点"""
            messages = state['messages']
            existing_summary = state.get("summary", "")

            # 计算当前消息 token 数
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            total_tokens = 0
            for msg in messages:
                content = msg.content if isinstance(msg.content, str) else ""
                total_tokens += len(encoding.encode(content))

            if total_tokens < max_tokens:
                return {}

            tokens = 0
            cut_index = 0
            for i, msg in enumerate(messages):
                content = msg.content if isinstance(msg.content, str) else ""
                tokens += len(encoding.encode(content))
                # 删减后的 token 满足限制
                if total_tokens - tokens < max_tokens:
                    cut_index = i + 1
                    break

            if cut_index < len(messages):
                first_kept_msg = messages[cut_index]
                if isinstance(first_kept_msg, ToolMessage):
                    cut_index += 1

            summary_msg = messages[:cut_index]
            delete_msg = [RemoveMessage(id=msg.id) for msg in summary_msg]

            summary_prompt = (
                "请将上面的对话内容总结为一个摘要。"
                f"现有的摘要：{existing_summary}"
            )
            summary_message = await llm.ainvoke(
                summary_msg + [HumanMessage(content=summary_prompt)],
                )
            new_summary = summary_message.content

            return {
                "messages": delete_msg,
                "summary": new_summary,
            }

        async def _agent_node(state: AgentState):
            messages = state["messages"]
            summary = state.get("summary", "")
            last_user_query = _last_user_text(messages)

            system_prompt = """你是一个高智能对话系统的**任务调度与决策中枢 (Central Orchestrator)**。

            【角色定位】
            你拥有多种专业工具的调用权限。你的核心职责不是机械地回复，而是作为**大脑**，分析用户意图，精准调度工具或调取记忆来解决问题。

            【核心原则：最新指令优先 (Priority on Latest Instruction)】
            在多轮对话中，用户意图经常会发生漂移（Intent Drift）。你必须严格遵守以下规则：
            1. **锚定当下**：无论之前的对话上下文多么长（如长篇写作、代码生成），你必须**优先响应用户最新发送的一条指令**。
            2. **打破惯性 (Break Context Inertia)**：
               - 严禁被上文的格式带偏。如果上文是写作文，而用户最新问“几点了”，立即切换回简短回答模式，**绝对不要**再写一篇作文。
               - 严禁在用户询问“回顾历史”时生成新内容。

            【决策逻辑与资源调度】
            请根据用户最新指令的性质，选择唯一的处理路径：
            - **路径 A：需要外部能力**（如事实查询、计算、实时信息）
              ➜ 必须调用对应的 **Tools**，严禁凭空猜测。
            - **路径 B：需要回顾历史**（如“我刚才说什么了”、“总结上文”）
              ➜ 调取 **对话历史 (Messages)** 或 **摘要 (Summary)** 进行事实复述。
            - **路径 C：纯逻辑/闲聊**（如打招呼、通用问答）
              ➜ 直接利用自身能力简练回复。

            【思维链 (Reasoning) 协议】
            在输出最终结果前，必须在 `reason` 字段中执行隐式推理：
            1. **意图判别**：用户的最新意图属于上述哪种路径（A/B/C）？
            2. **上下文清洗**：确认是否需要忽略上文的干扰信息（如长文本）？
            3. **工具决策**：如果需要调用工具，理由是什么？

            【执行任务优先】
            当用户要求你“执行任务”（例如运行测试、检查文件、读取配置、给出执行证据）时，必须优先调用可用工具完成任务，而不是仅口头解释。
            - 运行命令请优先以 dry-run 方式开始。
            - dry-run 仅代表“计划执行”，不是“实际执行成功”；除非 dry_run=false 且拿到真实 stdout/stderr，否则不得声称命令已成功运行。
            - 文件写入前必须明确拿到 confirm=true。
            - 若用户要求执行 git 提交流程，必须分三次调用命令：`git add .`、`git commit -m "..."`、`git push`；严禁使用 `&&` 串联成一条命令。
            - 最终回答需包含：结论、执行记录、证据、风险与回滚建议。
            
            【输出规范】
            1. **完整性原则**：如果用户要求生成长文本（作文、报告、代码），你必须生成用户需要的答案。
            2. **严禁偷懒**：不要因为是 JSON 格式就省略内容。

            请保持客观、冷静、服务型的对话风格。"""
            system_msg = [SystemMessage(content=system_prompt)]

            if summary:
                system_msg.append(SystemMessage(content=f"之前的对话摘要：{summary}"))

            if _is_recall_query(last_user_query):
                system_msg.append(
                    SystemMessage(
                        content=(
                            "检测到这是历史回顾问题。严禁调用任何外部工具（含RAG/搜索/命令执行）；"
                            "只能基于当前会话消息和摘要回答。若信息不足，明确说明缺失信息，不要编造。"
                        )
                    )
                )

            messages = system_msg + messages
            messages = _sanitize_tool_call_sequence(messages)

            result = await llm_with_tools.ainvoke(messages)
            return {"messages": [result]}

        async def _tool_node(state: AgentState):
            last_msg = state["messages"][-1]

            if not last_msg.tool_calls:
                return {}

            # 历史回顾类问题禁止调用工具，防止误触发 RAG/搜索造成答案漂移。
            last_user_query = _last_user_text(state["messages"])
            if _is_recall_query(last_user_query):
                deny_msgs = []
                for tool_call in last_msg.tool_calls:
                    spec = specs_by_name.get(tool_call.get("name"))
                    if spec and spec.policy.allow_in_recall:
                        continue
                    else:
                        deny_msgs.append(
                            ToolMessage(
                                content=(
                                    "Error: 该问题属于会话历史回顾，禁止调用外部工具。"
                                    "请直接依据 messages/summary 回答。"
                                ),
                                tool_call_id=tool_call["id"],
                            )
                        )
                if deny_msgs:
                    return {"messages": deny_msgs}

            tool_msgs: list[ToolMessage] = []
            for tool_call in last_msg.tool_calls:
                name = tool_call.get("name")
                args = tool_call.get("args", {})

                if name not in tools_by_name:
                    output = "Error: 调用不存在的工具"
                else:
                    try:
                        spec = specs_by_name.get(name)
                        if spec and spec.policy.require_confirm_param:
                            confirm_key = spec.policy.confirm_param_name
                            confirmed = bool(args.get(confirm_key)) if isinstance(args, dict) else False
                            if not confirmed:
                                interactive = await _human_confirm_tool_call(
                                    name,
                                    args if isinstance(args, dict) else {},
                                    confirm_key,
                                )
                                if interactive is True and isinstance(args, dict):
                                    args = dict(args)
                                    args[confirm_key] = True
                                elif interactive is False:
                                    output = (
                                        f"Error: 用户拒绝执行工具 {name}，已取消本次写操作。"
                                    )
                                    tool_msgs.append(
                                        ToolMessage(content=str(output), tool_call_id=tool_call["id"])
                                    )
                                    continue
                                else:
                                    output = (
                                        f"Error: 工具 {name} 需要显式 {confirm_key}=true，"
                                        "当前环境不支持交互确认，已拒绝执行。"
                                    )
                                    tool_msgs.append(
                                        ToolMessage(content=str(output), tool_call_id=tool_call["id"])
                                    )
                                    continue

                        tool_func = tools_by_name[name]
                        output = await tool_func.ainvoke(args)
                    except Exception as e:
                        output = f"Error: {e}"

                tool_msgs.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

            return {"messages": tool_msgs}

        graph = StateGraph(AgentState)

        graph.add_node("summary", _summary_node)
        graph.add_node("agent", _agent_node)
        graph.add_node("tools", _tool_node)

        graph.add_node("formatter", _structured_node)
        
        graph.set_entry_point("summary")
        
        graph.add_edge("summary", "agent")
        graph.add_edge("tools", "agent")

        def agent_continue(state: AgentState):
            last_msg = state["messages"][-1]
            if last_msg.tool_calls:
                return "tools"
            else:
                return "formatter"

        graph.add_conditional_edges("agent", agent_continue)
        graph.add_edge("formatter", "__end__")

        # 建立 Postgres 连接池
        # 连接字符串格式: postgresql://用户名:密码@地址:端口/数据库名
        # 例如: postgresql://postgres:123456@localhost:5432/agent_db
        db_url = os.getenv("POSTGRES_URL")

        conn_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        pool = AsyncConnectionPool(
            conninfo=db_url,
            max_size=20,
            kwargs=conn_kwargs,
            open=False,
        )

        await pool.open()

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()  # 第一次运行时，需要创建表结构

        compiled_graph = graph.compile(checkpointer=checkpointer)
        return cls(compiled_graph, pool)

    async def ainvoke(self, query: str, thread_id: str = None):
        """
        封装后的调用接口
        :param query: 用户的纯文本问题
        :param thread_id: 会话 ID，用于记忆隔离
        :return: 最终的结构化结果 (Receipt 对象) 或 错误信息
        """
        def _enforce_dry_run_truth(user_query: str, answer_text: str) -> str:
            q = (user_query or "").lower()
            if "dry-run" not in q and "dry run" not in q and "dry_run" not in q:
                return answer_text
            risky_patterns = [r"运行成功", r"没有报错", r"测试通过", r"成功运行", r"全部通过"]
            if any(re.search(p, answer_text) for p in risky_patterns):
                return (
                    "本次仅执行了 dry-run（计划演练），命令未实际运行，因此不能得出“测试通过/无报错”的结论。"
                    "如需真实结果，请在确认后执行非 dry-run 的 pytest -q。"
                )
            return answer_text

        # 回顾类问题改为“让模型基于记忆回答”，而不是只返回上一条缓存。
        # 这里把本地缓存作为候选记忆注入，让模型综合 checkpoint 历史 + 摘要 + 候选结论给出回答。
        if thread_id and _is_recall_query(query):
            cached_answers = self.thread_answer_cache.get(thread_id, [])
            if cached_answers:
                recent = cached_answers[-8:]
                memory_lines = "\n".join(f"- 记忆{i+1}: {item}" for i, item in enumerate(recent))
                query = (
                    f"{query}\n\n"
                    "以下是当前 thread 的候选记忆，请你基于完整会话上下文与这些候选信息进行回顾，"
                    "不要只机械复述最后一条：\n"
                    f"{memory_lines}"
                )

        inputs = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None

        # 执行图
        final_state = await self.runnable.ainvoke(inputs, config=config)

        # 优先返回结构化答案，如果没有（比如出错了），返回最后一条文本消息
        if final_state.get("structured_answer"):
            receipt = final_state["structured_answer"]
            receipt.answer = _enforce_dry_run_truth(query, receipt.answer)
            if any(k in receipt.answer for k in ["请在本地终端中运行", "请运行以下命令", "重新确认执行"]):
                for msg in reversed(final_state.get("messages", [])):
                    if not isinstance(msg, ToolMessage):
                        continue
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if "executed_command" not in content:
                        continue
                    try:
                        data = ast.literal_eval(content)
                    except Exception:
                        continue
                    if isinstance(data, dict) and data.get("executed"):
                        cmd = data.get("executed_command") or data.get("command")
                        rc = data.get("returncode")
                        hint = data.get("hint") or ""
                        receipt.answer = f"命令已执行：{cmd}；returncode={rc}。{hint}".strip()
                        break
            if thread_id:
                self.thread_answer_cache.setdefault(thread_id, []).append(receipt.answer)
            return receipt
        else:
            text = final_state["messages"][-1].content
            text = _enforce_dry_run_truth(query, str(text))
            if any(k in str(text) for k in ["请在本地终端中运行", "请运行以下命令", "重新确认执行"]):
                for msg in reversed(final_state.get("messages", [])):
                    if not isinstance(msg, ToolMessage):
                        continue
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if "executed_command" not in content:
                        continue
                    try:
                        data = ast.literal_eval(content)
                    except Exception:
                        continue
                    if isinstance(data, dict) and data.get("executed"):
                        cmd = data.get("executed_command") or data.get("command")
                        rc = data.get("returncode")
                        hint = data.get("hint") or ""
                        text = f"命令已执行：{cmd}；returncode={rc}。{hint}".strip()
                        break
            if thread_id:
                self.thread_answer_cache.setdefault(thread_id, []).append(str(text))
            return text

    async def aclose(self):
        await self.pool.close()
