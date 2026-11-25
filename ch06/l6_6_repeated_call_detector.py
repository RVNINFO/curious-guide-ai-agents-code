"""Listing 6.6 - A repeated-call detector for the agent in Section 5.11.1.

Stops a run that keeps making the same tool call with the same arguments.
"""
# Repeated-call detector for the agent in Section 5.11.1.

import json
from collections import Counter
from langchain.agents.middleware import before_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

MAX_REPEATS = 3


@before_model(can_jump_to=["end"])
def stop_repeated_calls(state: AgentState, runtime: Runtime):
    calls = Counter(
        (c["name"], json.dumps(c["args"], sort_keys=True, default=str))
        for m in state["messages"]
        for c in (getattr(m, "tool_calls", None) or []))
    if calls and max(calls.values()) >= MAX_REPEATS:
        (name, _), _ = calls.most_common(1)[0]
        return {"messages": [AIMessage(
                    f"I stopped because {name} was called {MAX_REPEATS} "
                    "times with the same input. A person will take over.")],
                "jump_to": "end"}
    return None
