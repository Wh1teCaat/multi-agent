import asyncio
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any


def _load_agent_class() -> Any:
    """从同目录的 agent.py 动态加载 Agent，兼容 `python -m agent` 与脚本执行。"""
    current_dir = Path(__file__).resolve().parent
    agent_file = current_dir / "agent.py"

    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    spec = importlib.util.spec_from_file_location("agent_runtime_entry", agent_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 agent.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Agent"):
        raise RuntimeError("agent.py 中未找到 Agent 类")

    return module.Agent


def _extract_command(text: str) -> str | None:
    """从回答中提取可执行命令（优先 bash 代码块）。"""
    if not text:
        return None

    block = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", text, flags=re.S | re.I)
    if block:
        for line in block.group(1).splitlines():
            cmd = line.strip()
            if cmd and not cmd.startswith("#"):
                return cmd

    inline = re.search(r"((?:python3?|pytest|go\s+test|git)\s+[^\n`]+)", text, flags=re.I)
    if inline:
        return inline.group(1).strip()

    return None


async def _run_ctl() -> None:
    Agent = _load_agent_class()
    agent = await Agent.create()
    thread_id = os.getenv("AGENT_THREAD_ID", "ctl-test")

    print("[ctl] 已启动。输入需求并回车；输入 exit 退出。")
    print(f"[ctl] 当前 thread_id: {thread_id}")

    pending_command: str | None = None

    try:
        while True:
            try:
                user_input = await asyncio.to_thread(input, "you> ")
            except EOFError:
                print("\n[ctl] 收到 EOF，退出。")
                break

            query = user_input.strip()
            if not query:
                continue

            if query.lower() == "exit":
                print("[ctl] 已退出。")
                break

            normalized = query.lower()
            if pending_command and normalized in {"y", "yes", "执行", "确认", "帮我执行"}:
                query = (
                    f"请调用 run_command 执行命令：{pending_command}。"
                    "要求 dry_run=false、visible_in_terminal=true，并返回 returncode、stdout、stderr 证据。"
                )
                pending_command = None
            elif pending_command and ("帮我执行" in query or "执行一下" in query):
                query = (
                    f"请调用 run_command 执行命令：{pending_command}。"
                    "要求 dry_run=false、visible_in_terminal=true，并返回 returncode、stdout、stderr 证据。"
                )
                pending_command = None

            try:
                result = await agent.ainvoke(query, thread_id)
                answer = result.answer if hasattr(result, "answer") else str(result)
                print(f"agent> {answer}")

                extracted = _extract_command(answer)
                if extracted:
                    pending_command = extracted
            except Exception as e:
                print(f"[ctl] 执行失败: {e}")
    except KeyboardInterrupt:
        print("\n[ctl] 用户中断，退出。")
    finally:
        await agent.aclose()


def main() -> None:
    asyncio.run(_run_ctl())


if __name__ == "__main__":
    main()
