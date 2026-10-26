"""Listing 5.x - Tenant-scoped refund tool: authority comes from runtime context, not the model.

Book: Part 5, tool-authorization example (ToolRuntime + context_schema).
Run offline:  python -m ch05.l5_tenant_scoped_refund
"""
from dataclasses import dataclass
from decimal import Decimal

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from ch05.adapters import db, payments


@dataclass
class AppContext:
    user_id: str
    tenant_id: str
    refund_operation_id: str


class RefundArgs(BaseModel):
    order_id: str
    amount: Decimal = Field(gt=0, decimal_places=2)
    reason: str


# `db` and `payments` are application-owned adapters (ch05/adapters.py), not framework APIs.
# Their queries must enforce tenant scope directly.

@tool(args_schema=RefundArgs)
def issue_refund(
    order_id: str,
    amount: Decimal,
    reason: str,
    runtime: ToolRuntime[AppContext],
) -> dict:
    """Issue a refund. Amounts over ₹5,000 require escalation."""

    tenant_id = runtime.context.tenant_id
    user_id = runtime.context.user_id

    # Never fetch the order globally and filter afterward.
    order = db.orders.get_for_tenant(
        tenant_id=tenant_id,
        order_id=order_id,
    )

    if order is None:
        return {"error": "Order not found or not accessible."}

    if order.user_id != user_id:
        return {"error": "This order does not belong to this customer."}

    # Refund history must obey the same tenant boundary.
    already = db.refunds.total_for_order(
        tenant_id=tenant_id,
        order_id=order_id,
    )

    if amount + already > order.total:
        return {
            "error": f"Only {order.total - already} remains refundable."
        }

    if amount > Decimal("5000"):
        return {
            "error": "Refunds over ₹5,000 require a supervisor. "
                     "Escalate this to a human."
        }

    return payments.refund(
        tenant_id=tenant_id,
        order_id=order_id,
        amount=amount,
        reason=reason,
        idempotency_key=(
            f"refund:{tenant_id}:"
            f"{runtime.context.refund_operation_id}"
        ),
    )


def build_agent(model="anthropic:claude-sonnet-4-6"):
    return create_agent(
        model=model,
        tools=[issue_refund],
        context_schema=AppContext,
    )


def run(agent, text: str, ctx: AppContext):
    return agent.invoke({"messages": [("user", text)]}, context=ctx)


if __name__ == "__main__":
    from ch05.adapters import Order, reset
    from common.fakes import ScriptedChatModel, ai

    reset([Order("ORD-1", "tenant-a", "u_1120", Decimal("800"))])
    m = ScriptedChatModel(script=[
        ai("", ("issue_refund", {"order_id": "ORD-1", "amount": "800", "reason": "damaged"})),
        ai("Done."),
    ])
    out = run(build_agent(m), "Refund ORD-1",
              AppContext("u_1120", "tenant-a", "refund-op-8721"))
    print(out["messages"][-2].content)
