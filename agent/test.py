import asyncio
from agent import Agent


async def main():
    agent = await Agent.create()
    res = await agent.ainvoke(
        "请执行发布前检查：先dry-run运行 pytest -q，再给出结论。",
        "task1",
    )
    print("[task1-dry-run]", res.answer)

    # 第二步：真实执行并在终端可见
    res = await agent.ainvoke(
        "请调用 run_command 执行 pytest -q，要求 dry_run=false、visible_in_terminal=true，然后给出结论。",
        "task1",
    )
    print("[task1-real-run]", res.answer)

    res = await agent.ainvoke("帮我把pip这个包更新一下", "task2")
    print("[task2]", res.answer)

    res = await agent.ainvoke("刚刚 task1 的结论是什么", "task1")
    print("[task1-memory]", res.answer)

    res = await agent.ainvoke("刚刚 task2 的结论是什么", "task2")
    print("[task2-memory]", res.answer)

    await agent.aclose()


if __name__ == '__main__':
    asyncio.run(main())