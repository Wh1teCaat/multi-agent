import asyncio
from agent import Agent

async def main():
    agent = await Agent.create()
    res = await agent.ainvoke("什么是langgraph",  "test1")
    print(res.answer)
    res = await agent.ainvoke("2025.12.1武汉天气", "test2")
    print(res.answer)
    res = await agent.ainvoke("刚刚问了什么问题", "test1")
    print(res.answer)
    res = await agent.ainvoke("刚刚问了什么问题", "test2")
    print(res.answer)

    await agent.aclose()

if __name__ == '__main__':
    asyncio.run(main())