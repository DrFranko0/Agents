import asyncio
from agent import agentic_flow

async def main():
    output = await agentic_flow.start({
        "latest_user_message": "I want an AI agent to help me with Framer Motion animations.",
        "messages": [],
        "scope": ""
    })
    print("Final output:", output)

if __name__ == "__main__":
    asyncio.run(main())