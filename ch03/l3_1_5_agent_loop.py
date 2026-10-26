"""Listing 3.1.5 - The agent loop, framework-free.

Book: Part 3, section 3.1.5 "Agent loops".
The book prints this as pseudocode; here the assumed names (SYSTEM_PROMPT, valid,
allowed, logger) are defined so it runs. Run:  python -m ch03.l3_1_5_agent_loop
"""
import logging

from common.mini_agent import Call, FakeModel, Response, Tool

logger = logging.getLogger("agent")
SYSTEM_PROMPT = "You are a helpful assistant. Use tools when needed."


def valid(arguments: dict, schema: dict) -> bool:
    """Minimal validator: arguments must be a dict containing every required key."""
    return isinstance(arguments, dict) and all(k in arguments for k in schema.get("required", []))


def allowed(tool, arguments: dict) -> bool:
    """Permission hook. Replace with a real policy check; here every registered tool is allowed."""
    return True


# --- the listing -----------------------------------------------------------
def run_agent(model, user_input: str, tools: list, max_steps: int = 15) -> str:
    tools_by_name = {t.name: t for t in tools}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    for _ in range(max_steps):
        response = model.chat(messages=messages, tools=[t.schema for t in tools])
        messages.append(response.message)
        if not response.tool_calls:
            return response.message.content  # finished
        for call in response.tool_calls:
            tool = tools_by_name.get(call.name)
            if tool is None:
                result = f"Error: no tool named {call.name}"
            elif not valid(call.arguments, tool.schema):  # validate
                result = "Error: invalid arguments"
            elif not allowed(tool, call.arguments):  # check permissions
                result = "Error: not permitted"
            else:
                try:
                    result = tool.run(**call.arguments)
                except Exception:
                    logger.exception("tool %s failed", call.name)
                    result = {
                        "error": "TOOL_FAILED",
                        "message": "The operation failed. Try a different approach or escalate.",
                    }
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return "I wasn't able to complete this within the step limit."


def main() -> None:
    model = FakeModel([
        Response(tool_calls=[Call("1", "add", {"a": 2, "b": 3})]),
        Response(content="2 + 3 = 5"),
    ])
    print(run_agent(model, "What is 2 + 3?", [Tool("add", lambda a, b: a + b)]))


if __name__ == "__main__":
    main()
