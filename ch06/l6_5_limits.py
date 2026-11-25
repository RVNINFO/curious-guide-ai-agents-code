"""Listing 6.5 - Hard limits for the agent in Section 5.11.1 (model calls, tool calls, per-tool cap).

build_agent in ch05.l5_11_1_capstone already applies the first two; this listing shows them on their own.
"""
# Hard limits for the agent in Section 5.11.1.
from langchain.agents.middleware import (
    ModelCallLimitMiddleware, ToolCallLimitMiddleware)

LIMITS = [
    # at most 12 model calls per run, then end cleanly
    ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),
    # at most 10 tool calls per run, across all tools;
    # extra calls are blocked with an error the model can read
    ToolCallLimitMiddleware(run_limit=10),
    # policy search rarely needs more than three tries
    ToolCallLimitMiddleware(tool_name="search_policy",
                            run_limit=3),
]

# In build_agent, put the limits first in the middleware list:
#     middleware=[*LIMITS, HumanInTheLoopMiddleware(...)],
