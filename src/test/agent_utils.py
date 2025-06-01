from typing import AsyncIterator


async def print_agent_response(agent_response_iter: AsyncIterator[dict]):
    tool_calls_dict = {}
    async for resp in agent_response_iter:
        if resp.get('agent'):
            messages = resp['agent']['messages']
            for message in messages:
                if message.additional_kwargs and message.additional_kwargs.get('tool_calls'):
                    for call in message.additional_kwargs.get('tool_calls'):
                        tool_calls_dict[call['id']] = call
                        # print(f"Tool call: {call['function']['name']}: {call['function']['arguments']}")
                else:
                    print(f"Agent: {message.content}")

        elif resp.get('tools'):
            tools = resp['tools']['messages']
            for tool in tools:
                call = tool_calls_dict[tool.tool_call_id]
                print(f"Tool: {call['function']['name']}({call['function']['arguments']})\n"
                      f"Return: {tool.content}")
        else:
            print(resp)
        print("------------------")