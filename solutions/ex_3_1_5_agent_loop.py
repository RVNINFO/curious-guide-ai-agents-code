"""Reference solution for the Part 3 exercise (agent loop)."""
from exercises.ex_3_1_5_agent_loop_starter import Call, FakeModel, Response, Tool  # noqa: F401

STEP_LIMIT_MESSAGE = "I wasn't able to complete this within the step limit."


def run_agent(model, user_input, tools, max_steps=5):
    tools_by_name = {t.name: t for t in tools}
    messages = [{"role": "user", "content": user_input}]
    for _ in range(max_steps):
        response = model.chat(messages, [t.schema for t in tools])
        messages.append({"role": "assistant", "content": response.content})
        if not response.tool_calls:
            return response.content
        for call in response.tool_calls:
            tool = tools_by_name.get(call.name)
            if tool is None:
                result = f"Error: no tool named {call.name}"
            else:
                try:
                    result = tool.run(**call.arguments)
                except Exception as exc:  # a failing tool must not crash the loop
                    result = f"Error: tool {call.name} failed ({type(exc).__name__})"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return STEP_LIMIT_MESSAGE
