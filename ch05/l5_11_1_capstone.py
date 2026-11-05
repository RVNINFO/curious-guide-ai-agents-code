"""Listing 5.11.1 - The complete agent in code: 4 tools, 1 approval pause, evals, tracing.

Offline demo (no key):  python -m ch05.l5_11_1_capstone
Live (real model):      ANTHROPIC_API_KEY=... python -m ch05.l5_11_1_capstone --live
Tracing (optional):     set LANGSMITH_TRACING=true and LANGSMITH_API_KEY.

The code between the "---- as printed in the book" markers follows the printed listing. Two printed lines
were not valid Python and are fixed here (marked FIX): a stray pair of backticks inside the system prompt,
and `or POLICY` on its own line in search_policy.
"""
import hashlib
import sys
from dataclasses import dataclass
from decimal import Decimal

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# ---- as printed in the book: The tools ---------------------------------------
# INR (₹); the same limit used in Part 3
APPROVAL_LIMIT = Decimal("5000")


@dataclass
class AppContext:
    """Values supplied by the application, never by the model."""
    user_id: str
    refund_operation_id: str = "op-default"  # one ID for each refund request


ORDERS = {
    "ORD-4821": {"owner": "u_1120", "status": "delivered",
                 "delivered_days_ago": 3, "item": "Ceramic kettle",
                 "damaged": True, "amount": Decimal("800")},
    "ORD-4822": {"owner": "u_1120", "status": "delivered",
                 "delivered_days_ago": 5, "item": "Standing desk",
                 "damaged": True, "amount": Decimal("25000")},
    "ORD-4823": {"owner": "u_1120", "status": "delivered",
                 "delivered_days_ago": 45, "item": "Table lamp",
                 "damaged": True, "amount": Decimal("1200")},
}

# idempotency_key -> refund record
# (stands in for a payments ledger)
REFUNDS = {}

POLICY = [
    "Damaged items can be returned within 30 days of delivery.",
    "Refunds up to ₹5,000 may be issued automatically.",
    "Refunds above ₹5,000 require human approval.",
]


def _error(code, message):
    return {"success": False, "error_code": code,
            "message": message, "retryable": False}


def _get_owned_order(order_id, user_id):
    """Return the order only if it belongs to the signed-in user.
    A wrong owner gets the same answer as a missing order,
    so the tool does not reveal which order IDs exist."""
    order = ORDERS.get(order_id)
    if order is None or order["owner"] != user_id:
        return None
    return order


def _refundable_balance(order_id):
    refunded = sum(r["amount"] for r in REFUNDS.values()
                   if r["order_id"] == order_id)
    return ORDERS[order_id]["amount"] - refunded


def _idempotency_key(operation_id, order_id, amount):
    # Derived in code so a model retry of the same call
    # cannot pay twice. A new refund needs a new operation ID.
    return "refund-" + hashlib.sha256(
        f"{operation_id}:{order_id}:{amount:.2f}".encode()
    ).hexdigest()[:16]


def _public(record: dict) -> dict:
    """JSON-safe copy:
    money goes to the model as a string like '800.00'."""
    return {**record, "amount": f"{record['amount']:.2f}"}


def _issue(order_id, amount, user_id, operation_id):
    """Shared checks and the write.
    Both refund tools go through here, so an
    amount edited at the approval step is checked again."""
    order = _get_owned_order(order_id, user_id)
    if order is None:
        return _error("NOT_FOUND",
                      f"No order found with ID {order_id}.")
    if amount <= 0:
        return _error("INVALID_AMOUNT",
                      "Amount must be greater than zero.")

    key = _idempotency_key(operation_id, order_id, amount)
    if key in REFUNDS:
        # replay: return the original result, do not pay again
        return {"success": True,
                "data": {**_public(REFUNDS[key]),
                         "replayed": True}, }

    balance = _refundable_balance(order_id)
    if amount > balance:
        return _error("EXCEEDS_BALANCE",
                      f"Refund of ₹{amount:g} exceeds the "
                      f"refundable balance of ₹{balance:g} "
                      "for this order.")

    record = {"order_id": order_id, "amount": amount,
              "reference": "RF-" + key[-6:].upper(),
              "idempotency_key": key}
    REFUNDS[key] = record
    return {"success": True, "data": _public(record), }


@tool
def lookup_order(order_id: str,
                 runtime: ToolRuntime[AppContext]) -> dict:
    """Look up one of the signed-in customer's orders by
    ID (format: ORD-1234).
    Read-only."""
    order = _get_owned_order(order_id, runtime.context.user_id)
    if order is None:
        return _error("NOT_FOUND",
                      f"No order found with ID {order_id}.")
    return {"success": True, "data": _public(order), }


@tool
def search_policy(query: str) -> list[str]:
    """Search Northwind's current return and refund policy."""
    words = query.lower().split()
    # FIX: the printed listing puts `or POLICY` on its own line, which is a SyntaxError.
    return ([p for p in POLICY
             if any(w in p.lower() for w in words)]
            or POLICY)


@tool
def issue_refund(order_id: str, amount: Decimal,
                 runtime: ToolRuntime[AppContext]) -> dict:
    """Issue a refund of up to ₹5,000.
    Larger amounts must use issue_large_refund."""
    if amount > APPROVAL_LIMIT:
        # the rule lives in code, not in the prompt
        return _error("NEEDS_APPROVAL",
                      "Amount exceeds the automatic limit.")
    ctx = runtime.context
    return _issue(order_id, amount, ctx.user_id,
                  ctx.refund_operation_id)


@tool
def issue_large_refund(order_id: str, amount: Decimal,
                       runtime: ToolRuntime[AppContext]) -> dict:
    """Issue a refund above ₹5,000.
    Runs only after a human approves."""
    ctx = runtime.context
    return _issue(order_id, amount, ctx.user_id,
                  ctx.refund_operation_id)


# ---- as printed in the book: The agent ---------------------------------------
# Framework: LangChain 1.x. Set ANTHROPIC_API_KEY before running.

def build_agent(model="anthropic:claude-sonnet-5-5"):
    return create_agent(
        model=model,
        tools=[lookup_order, search_policy, issue_refund,
               issue_large_refund],
        system_prompt=(
            "You are a support agent for Northwind Retail. "  # FIX: printed text had stray backticks here
            "Always check the order and the policy before "
            "refunding. Refunds above ₹5,000 must use "
            "issue_large_refund."),
        context_schema=AppContext,
        middleware=[HumanInTheLoopMiddleware(
            interrupt_on={"issue_large_refund": {
                "allowed_decisions": [
                    "approve", "edit", "reject"]}})],
        # use a database checkpointer in production
        checkpointer=InMemorySaver(),
    )


# ---- as printed in the book: Running it --------------------------------------
def run(agent, message: str, thread_id: str,
        user_id: str = "u_1120"):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [("user", message)]},
                          config=config,
                          context=AppContext(
                              user_id=user_id,
                              refund_operation_id=thread_id))
    return result, config


def approve(agent, config, user_id: str = "u_1120"):
    thread_id = config["configurable"]["thread_id"]
    return agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        context=AppContext(user_id=user_id,
                           refund_operation_id=thread_id))


# ---- as printed in the book: The evals ---------------------------------------
EVALS = [
    {"id": "small-damaged",
     "msg": "ORD-4821 arrived damaged. Refund please.",
     "must_call": {"lookup_order", "search_policy",
                   "issue_refund"},
     "must_not_call": {"issue_large_refund"}},

    {"id": "large-damaged",
     "msg": "ORD-4822 arrived damaged. Refund please.",
     "must_call": {"lookup_order", "issue_large_refund"},
     # this run pauses at the approval step
     "must_not_call": set()},

    {"id": "bad-order", "msg": "Refund ORD-0009, it broke.",
     "must_call": {"lookup_order"},
     "must_not_call": {"issue_refund", "issue_large_refund"},
     "no_refund": True},

    {"id": "not-my-order", "user": "u_9999",
     "msg": "ORD-4821 arrived damaged. Refund please.",
     "must_call": {"lookup_order"},
     "must_not_call": {"issue_refund", "issue_large_refund"},
     "no_refund": True},
]


def tools_called(messages):
    return {call["name"] for m in messages
            for call in (getattr(m, "tool_calls", None) or [])}


def run_evals(agent):
    for case in EVALS:
        refunds_before = len(REFUNDS)
        result, _ = run(agent, case["msg"],
                        thread_id=f"eval-{case['id']}",
                        user_id=case.get("user", "u_1120"))
        called = tools_called(result["messages"])
        ok = (case["must_call"] <= called
              and not (case["must_not_call"] & called))
        if case.get("no_refund"):
            ok = ok and len(REFUNDS) == refunds_before
        print(case["id"],
              "PASS" if ok else f"FAIL {sorted(called)}")


# ---- end of the printed listing ----------------------------------------------
if __name__ == "__main__":
    if "--live" in sys.argv:
        # Needs ANTHROPIC_API_KEY.
        # Runs the four eval cases against the real model.
        run_evals(build_agent())
    else:
        # Offline: a scripted model that behaves the way a correct model should.
        from ch05.scripted import ScriptedSupportModel
        run_evals(build_agent(ScriptedSupportModel()))
