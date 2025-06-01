
import asyncio
import os

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters, ClientSession

from agent_utils import print_agent_response
from mcp.client.stdio import stdio_client


os.environ["PYTHONUNBUFFERED"] = "1"

from langgraph.prebuilt import create_react_agent

os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4-turbo"

with open("/Users/leqiuhong/PycharmProjects/langchain-mcp-adapters/tests/mock/sale_table.csv", "r") as f:
    sale_data = f.read()


# @pytest.mark.asyncio
async def atest_my_client():
    print("------------------")
    server_params = StdioServerParameters(
        command="uv",
        # Make sure to update to the full absolute path to your math_server.py file
        args=["--directory",
              "/Users/leqiuhong/PycharmProjects/mcp-server-data-exploration/src/mcp_server_ds",
              "run",
              "mcp-server-ds"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            # tools = await client.get_tools()
            print("------------------")
            agent = create_react_agent("azure_openai:gpt-4o", tools)
            user_message = (
                f"不需要read_csv, 现在已经有了DataFrame数据: sale_table\n"
                f"然后，你需要根据用户的问题来分析数据，结合数据和你自己的分析结果来回答问题\n"
                f"问题：\n"
                f"异常洞察")
            agent_response_iter = agent.astream({"messages": user_message})

            await print_agent_response(agent_response_iter)


if __name__ == '__main__':
    asyncio.run(atest_my_client())
