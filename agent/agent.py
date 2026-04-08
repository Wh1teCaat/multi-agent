import ast
import os
import re
from typing import Any, TypedDict, Annotated, Optional

from langgraph.types import interrupt, Command
from langgraph.errors import GraphInterrupt

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
from tools.command_runtime import requires_second_confirmation
from skills import SkillPrompts

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
    """结构化输出（含 ReAct 闭环过程）"""
    reason: str = Field(
        default="",
        description="总体推理结论：说明意图判别、路径选择和为什么这样做。"
    )
    thought: str = Field(
        default="",
        description="行动前推理：当前子目标、信息缺口、是否需要工具。"
    )
    action: str = Field(
        default="",
        description="执行动作：调用了什么工具或采取了什么具体步骤。"
    )
    observation: str = Field(
        default="",
        description="行动观察：工具返回了什么证据、报错或状态变化。"
    )
    reflection: str = Field(
        default="",
        description="行动后反思：基于 observation 如何修正后续决策。"
    )
    answer: str = Field(
        description="针对用户问题的最终回答内容；若用户要求长文本/代码，必须给出完整内容。"
    )
    source: list[str] = Field(default_factory=list, description="回答中引用的具体文档名称或页码列表。如果没用到文档，请留空。")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: Optional[str]
    structured_answer: Optional[Receipt]


class Agent:
    def __init__(self, runnable, pool):
        self.runnable = runnable
        self.pool = pool
        self.thread_answer_cache: dict[str, list[str]] = {}

    async def aresume(self, thread_id: str, resume_value: Any):
        return await self.ainvoke(Command(resume=resume_value), thread_id)

    @classmethod
    async def create(cls, max_tokens=5000):
        max_tokens = max_tokens
        tool_specs = build_default_tool_specs()
        tools, tools_by_name, specs_by_name = build_tool_maps(tool_specs)
        llm = ChatOpenAI(model=os.getenv("MODEL_NAME"))
        llm_with_tools = llm.bind_tools(tools, tool_choice="auto")
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

        def _approval_from_resume(resume_value: Any) -> bool:
            """解析 interrupt 的 resume 值，True 为批准，False 为拒绝。"""
            if isinstance(resume_value, bool):
                return resume_value
            if isinstance(resume_value, str):
                normalized = resume_value.strip().lower()
                return normalized in {"y", "yes", "true", "1", "ok", "approve", "approved", "允许", "同意", "是", "批准"}
            if isinstance(resume_value, dict):
                raw = resume_value.get("approved", resume_value.get("confirm", resume_value.get("decision")))
                return _approval_from_resume(raw)
            return False


        async def _structured_node(state: AgentState):
            messages = state["messages"]
            summary = state.get("summary", "")
            messages = _sanitize_tool_call_sequence(messages)

            skill_system = [
                SystemMessage(content=SkillPrompts.ORCHESTRATION),
                SystemMessage(content=SkillPrompts.INTENT_PRIORITY),
                SystemMessage(content=SkillPrompts.REACT_LOOP),
                SystemMessage(content=SkillPrompts.STRUCTURED_OUTPUT_POLICY),
                SystemMessage(content=SkillPrompts.TOOL_POLICY),
                SystemMessage(content=SkillPrompts.OUTPUT_POLICY),
                SystemMessage(content="若上文存在 run_command 的工具回执，必须严格基于回执给出结论；禁止让用户再次手工去本地执行同一命令。"),
            ]

            if summary:
                prompt_msg = [SystemMessage(content=f"上下文摘要：{summary}")] + skill_system + messages
            else:
                prompt_msg = skill_system + messages

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
            messages = _sanitize_tool_call_sequence(messages)

            prompt_msg: list[BaseMessage]
            base_system = [
                SystemMessage(content=SkillPrompts.ORCHESTRATION),
                SystemMessage(content=SkillPrompts.INTENT_PRIORITY),
                SystemMessage(content=SkillPrompts.REACT_LOOP),
                SystemMessage(content=SkillPrompts.TOOL_POLICY),
                SystemMessage(content=SkillPrompts.EXECUTION_POLICY),
                SystemMessage(content=SkillPrompts.OUTPUT_POLICY),
                SystemMessage(content="若上文存在 run_command 的工具回执，必须严格基于回执给出结论；禁止让用户再次手工去本地执行同一命令。"),
            ]

            if summary:
                prompt_msg = [SystemMessage(content=f"上下文摘要：{summary}")] + base_system + messages
            else:
                prompt_msg = base_system + messages

            response = await llm_with_tools.ainvoke(prompt_msg)
            return {"messages": [response]}


        async def _tool_node(state: AgentState):
            last_msg = state["messages"][-1]

            tool_calls = getattr(last_msg, "tool_calls", None)
            if not tool_calls:
                return {}

            # 历史回顾类问题禁止调用工具，防止误触发 RAG/搜索造成答案漂移。
            last_user_query = _last_user_text(state["messages"])
            if _is_recall_query(last_user_query):
                deny_msgs = []
                for tool_call in tool_calls:
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

            def _should_interrupt_tool_call(tool_name: str, tool_args: Any) -> tuple[bool, dict[str, Any]]:
                """工具调用安全闸：
                - 写入类工具：统一中断等待人工确认
                - run_command：命令需要二次确认（或显式请求 confirm）时中断
                """
                spec = specs_by_name.get(tool_name)
                args_dict: dict[str, Any] = tool_args if isinstance(tool_args, dict) else {}

                # 统一：注册表声明为 high side-effect 的工具，默认要求人工确认
                if spec and spec.policy.side_effect_level == "high":
                    # run_command 属于 high，但我们对它做更细粒度判断，避免每次 ls 都中断
                    if tool_name != "run_command":
                        return True, {
                            "reason": "该工具具有写入/副作用能力（side_effect_level=high），需人工二次确认。",
                            "policy": {"side_effect_level": spec.policy.side_effect_level},
                        }
 
                if tool_name == "run_command":
                    command_text = str(args_dict.get("command", "")).strip()
                    requested_confirm = bool(args_dict.get("confirm", False))

                    if requested_confirm:
                        return True, {"reason": "命令请求以 confirm=true 执行（具备副作用），需人工二次确认。"}

                    if requires_second_confirmation(command_text):
                        return True, {"reason": "命令命中二次确认策略（高风险/副作用），需人工二次确认。"}

                return False, {}

            tool_msgs: list[ToolMessage] = []
            for tool_call in tool_calls:
                name = tool_call.get("name")
                args = tool_call.get("args", {})

                if name not in tools_by_name:
                    output = "Error: 调用不存在的工具"
                else:
                    try:
                        should_interrupt, risk = _should_interrupt_tool_call(name, args)
                        if should_interrupt:
                            resume_value = interrupt(
                                {
                                    "type": "human_approval_required",
                                    "tool_name": name,
                                    "args": args if isinstance(args, dict) else {},
                                    "message": "工具为高风险操作，等待人工审批。",
                                    "risk": risk,
                                    "recovery_hint": "可在终端使用相同 thread_id 恢复会话并输入 yes/no 继续。",
                                }
                            )
                            if not _approval_from_resume(resume_value):
                                output = f"Error: 用户拒绝执行工具 {name}，本次工具调用失败。"
                                tool_msgs.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
                                continue

                        tool_func = tools_by_name[name]
                        output = await tool_func.ainvoke(args)
                    except GraphInterrupt:
                        raise
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
            if getattr(last_msg, "tool_calls", None):
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

    async def ainvoke(self, query: str | Command, thread_id: str = None):
        """
        封装后的调用接口
        :param query: 用户的纯文本问题
        :param thread_id: 会话 ID，用于记忆隔离
        :return: 最终的结构化结果 (Receipt 对象) 或 错误信息
        """
        inputs: dict[str, Any] | Command
        if isinstance(query, Command):
            inputs = query
        else:
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

        def _interrupt_response(*, interrupts: Any = None, payload: Any = None):
            return {
                "__interrupt__": interrupts or True,
                "payload": payload,
                "thread_id": thread_id,
                "recovery": {
                    "can_resume_in_terminal": True,
                    "instruction": "使用相同 thread_id 在终端恢复，并输入 yes/no 进行审批。",
                },
            }

        # 执行图：不捕获 GraphInterrupt，统一通过返回值中的 __interrupt__ 判断中断
        final_state = await self.runnable.ainvoke(inputs, config=config)

        if final_state is None:
            return {"error": "graph returned None"}

        def _extract_interrupt_from_response(state: Any) -> tuple[Any, Any]:
            """从响应中提取中断信息（仅基于响应内容，不捕获异常）。"""
            if not isinstance(state, dict):
                return None, None

            # 1) 标准位置
            if "__interrupt__" in state:
                interrupts = state.get("__interrupt__")
                if isinstance(interrupts, (list, tuple)) and interrupts:
                    first = interrupts[0]
                    payload = getattr(first, "value", None) or first
                    return interrupts, payload
                if interrupts:
                    return interrupts, interrupts
                return True, None

            # 2) 兼容位置：有些实现会放在 interrupt/interrupts
            for key in ("interrupt", "interrupts"):
                if key in state and state.get(key):
                    interrupts = state.get(key)
                    if isinstance(interrupts, (list, tuple)) and interrupts:
                        first = interrupts[0]
                        payload = getattr(first, "value", None) or first
                        return interrupts, payload
                    return interrupts, interrupts

            # 3) 消息携带：AIMessage.additional_kwargs["__interrupt__"]
            messages = state.get("messages", [])
            if isinstance(messages, list):
                for msg in reversed(messages):
                    additional = getattr(msg, "additional_kwargs", None)
                    if isinstance(additional, dict) and "__interrupt__" in additional:
                        interrupts = additional.get("__interrupt__")
                        if isinstance(interrupts, (list, tuple)) and interrupts:
                            first = interrupts[0]
                            payload = getattr(first, "value", None) or first
                            return interrupts, payload
                        return interrupts or True, interrupts

            return None, None

        interrupts, payload = _extract_interrupt_from_response(final_state)
        if interrupts is not None:
            return _interrupt_response(interrupts=interrupts, payload=payload)

        # 优先返回结构化答案，如果没有（比如出错了），返回最后一条文本消息
        if final_state.get("structured_answer"):
            receipt = final_state["structured_answer"]
            if isinstance(receipt, dict):
                answer_text = str(receipt.get("answer", ""))
            else:
                answer_text = str(getattr(receipt, "answer", ""))

            if any(k in answer_text for k in ["请在本地终端中运行", "请运行以下命令", "重新确认执行"]):
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
                        answer_text = f"命令已执行：{cmd}；returncode={rc}。{hint}".strip()
                        break

            if isinstance(receipt, dict):
                receipt["answer"] = answer_text
                final_receipt = receipt
            else:
                setattr(receipt, "answer", answer_text)
                final_receipt = receipt

            if thread_id:
                self.thread_answer_cache.setdefault(thread_id, []).append(answer_text)
            return final_receipt
        else:
            text = final_state["messages"][-1].content
            text = str(text)
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
