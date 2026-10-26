"""Listing 3.7.1 - Approval workflow with HumanInTheLoopMiddleware + a checkpointer.

Run (offline demo):  python -m ch03.l3_7_1_human_in_the_loop
"""
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from ch03.l3_2_3_order_status_tool import get_order_status
from langchain_core.tools import tool

EXECUTED: list[tuple] = []  # records side effects so tests can prove what ran


@tool
def issue_refund(order_id: str, amount: float) -> dict:
    """Issue a refund for an order."""
    EXECUTED.append(("issue_refund", order_id, amount))
    return {"success": True, "order_id": order_id, "amount": amount}


@tool
def send_email(to: str, body: str) -> dict:
    """Send an email to a customer."""
    EXECUTED.append(("send_email", to))
    return {"success": True}


def build_agent(model="anthropic:claude-sonnet-4-6"):
    return create_agent(
        model=model,
        tools=[get_order_status, issue_refund, send_email],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "issue_refund": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                    },
                    "send_email": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                },
            ),
        ],
        checkpointer=InMemorySaver(),  # use a database in production
    )


if __name__ == "__main__":
    from langgraph.types import Command
    from common.fakes import ScriptedChatModel, ai

    model = ScriptedChatModel(script=[
        ai("", ("issue_refund", {"order_id": "ORD-1", "amount": 100.0})),
        ai("Refund issued."),
    ])
    agent = build_agent(model)
    cfg = {"configurable": {"thread_id": "demo"}}
    out = agent.invoke({"messages": [("user", "Refund ORD-1")]}, config=cfg)
    print("paused:", "__interrupt__" in out, "executed so far:", EXECUTED)
    out = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=cfg)
    print(out["messages"][-1].content, "executed:", EXECUTED)
