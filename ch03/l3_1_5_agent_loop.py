"""Listing 3.1.5 - The agent loop, framework-free.

Book: Part 3, section 3.1.5 "Agent loops".
The book prints this as pseudocode; here the assumed names (SYSTEM_PROMPT, validate, allowed, logger)
are defined so it runs. `model` is passed in as an argument instead of being a global.
Run:  python -m ch03.l3_1_5_agent_loop
"""
import json
import logging

from common.mini_agent import Call, FakeModel, Response, Tool

logger = logging.getLogger("agent")
SYSTEM_PROMPT = "You are a helpful assistant. Use tools when needed."


def validate(arguments: dict, schema: dict) -> str | None:
    """Return a message the model can act on, or None if the arguments are fine.
    Minimal: checks required keys. Real validators also check types, formats and ranges."""
    if not isinstance(arguments, dict):
        return "Arguments must be an object."
    missing = [k for k in schema.get("required", []) if k not in arguments]
    return f"Missing required argument(s): {', '.join(missing)}." if missing else None


def allowed(tool, arguments: dict) -> bool:
    """Permission hook. Replace with a real policy check; here every registered tool is allowed."""
    return True


# --- the listing -----------------------------------------------------------
def run_agent(model, user_input: str, tools: list,
              max_steps: int = 15) -> str:
    tools_by_name = {t.name: t for t in tools}

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}]

    for step in range(max_steps):
        response = model.chat(messages, tools)
        if not response.tool_calls:
            return response.message.content

        messages.append(response.message)

        for call in response.tool_calls:
            tool = tools_by_name.get(call.name)
            problem = (validate(call.arguments, tool.schema)
                       if tool else None)

            if tool is None:
                names = ", ".join(tools_by_name)
                result = (f"Error: no tool named {call.name}.\n"
                          f"Available tools: {names}.")
            elif problem:  # validate
                # e.g. "The date must be in YYYY-MM-DD format.
                # You supplied 'March 15'."
                result = f"Error: invalid arguments. {problem}"
            elif not allowed(tool, call.arguments):
                # check permissions
                result = ("Error: this action is not permitted. "
                          "Tell the user it needs approval.")
            else:
                try:
                    result = tool.run(**call.arguments)
                except Exception:
                    logger.exception("tool %s failed", call.name)
                    msg = ("The operation failed. "
                           "Try a different approach "
                           "or escalate.")
                    result = {"error": "TOOL_FAILED",
                              "message": msg}

            # Send strings as-is; serialize structured results as
            # JSON, not a Python representation

            content = (result if isinstance(result, str)
                       else json.dumps(result, default=str))

            messages.append({"role": "tool",
                             "tool_call_id": call.id,
                             "content": content})

    return "I wasn't able to complete this within the step limit."


def main() -> None:
    model = FakeModel([
        Response(tool_calls=[Call("1", "add", {"a": 2, "b": 3})]),
        Response(content="2 + 3 = 5"),
    ])
    print(run_agent(model, "What is 2 + 3?", [Tool("add", lambda a, b: a + b)]))


if __name__ == "__main__":
    main()
