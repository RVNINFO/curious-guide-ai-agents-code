"""Listing 3.4.1 - Short-term memory via SummarizationMiddleware (requires langchain>=1.1).

Live:  ANTHROPIC_API_KEY=... python -m ch03.l3_4_1_summarization_agent
"""
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from ch03.l3_2_3_order_status_tool import get_order_status
from ch03.l3_3_1_tool_schema import search_orders


def build_agent(model="anthropic:claude-sonnet-4-6",
                summary_model="anthropic:claude-haiku-4-5",
                trigger_tokens: int = 4000, keep_messages: int = 6):
    return create_agent(
        model=model,
        tools=[get_order_status, search_orders],
        middleware=[
            SummarizationMiddleware(
                model=summary_model,                # cheap model for the summary
                trigger=("tokens", trigger_tokens),  # summarize past this point
                keep=("messages", keep_messages),    # keep the last N verbatim
            ),
        ],
    )


if __name__ == "__main__":
    agent = build_agent()
    out = agent.invoke({"messages": [("user", "Where is order ORD-1234?")]})
    print(out["messages"][-1].content)
