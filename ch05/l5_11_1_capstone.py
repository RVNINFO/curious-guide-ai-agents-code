"""Listing 5.11.1 - The complete agent in code: 4 tools, 1 approval pause, evals, tracing.

Offline demo (no key):  python -m ch05.l5_11_1_capstone
Live:                   ANTHROPIC_API_KEY=... python -m ch05.l5_11_1_capstone --live
Tracing (optional):     set LANGSMITH_TRACING=true and LANGSMITH_API_KEY.
"""
import sys

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# ---- The tools --------------------------------------------------------------
APPROVAL_LIMIT = 5000  # INR (₹); the same limit used in Part 3

ORDERS = {
    "ORD-4821": {"owner": "u_1120", "status": "delivered",
                 "delivered_days_ago": 3, "item": "Ceramic kettle",
                 "damaged": True, "amount": 800},
    "ORD-4822": {"owner": "u_1120", "status": "delivered",
                 "delivered_days_ago": 5, "item": "Standing desk",
                 "damaged": True, "amount": 25000},
}

POLICY = [
    "Damaged items can be returned within 30 days of delivery.",
    "Refunds up to ₹5,000 may be issued automatically.",
    "Refunds above ₹5,000 require human approval.",
]


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID (format: ORD-1234). Read-only."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"success": False, "error_code": "NOT_FOUND",
                "message": f"No order found with ID {order_id}.",
                "retryable": False}
    return {"success": True, "data": order}


@tool
def search_policy(query: str) -> list[str]:
    """Search Northwind's current return and refund policy."""
    words = query.lower().split()
    return [p for p in POLICY if any(w in p.lower() for w in words)] or POLICY


@tool
def issue_refund(order_id: str, amount: float) -> dict:
    """Issue a refund of up to ₹5,000. Larger amounts must use
    issue_large_refund."""
    if amount > APPROVAL_LIMIT:  # the rule lives in code, not in the prompt
        return {"success": False, "error_code": "NEEDS_APPROVAL",
                "message": "Amount exceeds the automatic limit.",
                "retryable": False}
    return {"success": True,
            "data": {"order_id": order_id, "amount": amount,
                     "reference": "RF-29381"}}


@tool
def issue_large_refund(order_id: str, amount: float) -> dict:
    """Issue a refund above ₹5,000. Runs only after a human approves."""
    return {"success": True,
            "data": {"order_id": order_id, "amount": amount,
                     "reference": "RF-29382"}}


# ---- The agent -------------------------------------------------------------
def build_agent(model="anthropic:claude-sonnet-4-6"):
    return create_agent(
        model=model,
        tools=[lookup_order, search_policy, issue_refund, issue_large_refund],
        system_prompt=(
            "You are a support agent for Northwind Retail. Always check the "
            "order and the policy before refunding. Refunds above ₹5,000 "
            "must use issue_large_refund."),
        middleware=[HumanInTheLoopMiddleware(
            interrupt_on={"issue_large_refund": {
                "allowed_decisions": ["approve", "edit", "reject"]}})],
        checkpointer=InMemorySaver(),  # use a database checkpointer in production
    )


# ---- Running it ------------------------------------------------------------
# A run is identified by its thread_id. If a person approves later, the same thread resumes.
def run(agent, message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [("user", message)]}, config=config)
    return result, config


def approve(agent, config):
    return agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config=config)


# ---- The evals -------------------------------------------------------------
EVALS = [
    {"id": "small-damaged", "msg": "ORD-4821 arrived damaged. Refund please.",
     "must_call": {"lookup_order", "search_policy", "issue_refund"},
     "must_not_call": {"issue_large_refund"}},
    {"id": "large-damaged", "msg": "ORD-4822 arrived damaged. Refund please.",
     "must_call": {"lookup_order", "issue_large_refund"},
     "must_not_call": set()},  # this run pauses at the approval step
    {"id": "bad-order", "msg": "Refund ORD-0009, it broke.",
     "must_call": {"lookup_order"},
     "must_not_call": {"issue_refund", "issue_large_refund"}},
]


def tools_called(messages):
    return {call["name"] for m in messages
            for call in (getattr(m, "tool_calls", None) or [])}


def run_evals(agent, evals=EVALS, agent_for_case=None):
    """Returns {case_id: bool}. `agent_for_case(case)` lets tests supply a scripted agent per case."""
    results = {}
    for case in evals:
        a = agent_for_case(case) if agent_for_case else agent
        result, _ = run(a, case["msg"], thread_id=f"eval-{case['id']}")
        called = tools_called(result["messages"])
        ok = (case["must_call"] <= called
              and not (case["must_not_call"] & called))
        print(case["id"], "PASS" if ok else f"FAIL {sorted(called)}")
        results[case["id"]] = ok
    return results


if __name__ == "__main__":
    if "--live" in sys.argv:
        agent = build_agent()
        run_evals(agent)
    else:
        from ch05.scripted import scripted_agent_for_case
        run_evals(None, agent_for_case=scripted_agent_for_case)
