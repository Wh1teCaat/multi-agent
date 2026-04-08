import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from langgraph.types import Command

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


async def _run_ctl() -> None:
    Agent = _load_agent_class()
    agent = await Agent.create()
    thread_id = os.getenv("AGENT_THREAD_ID", "test-agent")

    print("[ctl] 已启动。输入需求并回车；输入 exit 退出。")
    print(f"[ctl] 当前 thread_id: {thread_id}")

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

            try:
                # 连续处理中断：支持一个请求内多次审批（例如 git add/commit/push 链式执行）。
                result = await agent.ainvoke(query, thread_id)
                while isinstance(result, dict) and "__interrupt__" in result:
                    print(f"[ctl] 检测到中断: {result.get('payload')}")

                    while True:
                        decision = await asyncio.to_thread(input, "[ctl] 是否批准继续？(yes/no): ")
                        normalized_decision = decision.strip().lower()
                        allowed_tokens = {
                            "y", "yes", "n", "no", "true", "false", "1", "0", "ok", "approve", "approved", "deny",
                            "允许", "同意", "拒绝", "是", "否", "批准", "不批准",
                        }
                        if normalized_decision not in allowed_tokens:
                            print("[ctl] 仅接受 yes/no（或同义词）作为审批输入。")
                            continue
                        break

                    approved = normalized_decision in {
                        "y", "yes", "true", "1", "ok", "approve", "approved", "允许", "同意", "是", "批准"
                    }
                    result = await agent.ainvoke(Command(resume=approved), thread_id)

                if isinstance(result, dict):
                    if "answer" in result:
                        answer = str(result.get("answer"))
                    elif "error" in result:
                        answer = f"执行失败: {result.get('error')}"
                    else:
                        answer = str(result)
                else:
                    answer = getattr(result, "answer", str(result))
                print(f"agent> {answer}")

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
