"""The 'test harness' that Listing 5.x's cross-tenant test assumes ("your own test harness").

`run_agent(request, test_identity, fixtures)` runs the tenant-scoped refund agent against fixture
orders and summarises what happened, so security properties can be asserted as plain tests.
"""
from dataclasses import dataclass
from decimal import Decimal

from ch05 import adapters
from ch05.l5_tenant_scoped_refund import AppContext, build_agent
from common.fakes import ScriptedChatModel, ai, iter_tool_names


@dataclass
class RunResult:
    status: str
    tool_calls: list
    cross_tenant_records_returned: int
    final_text: str

    def tool_calls_named(self, name: str) -> int:
        return sum(1 for n in self.tool_calls if n == name)


def run_agent(request: str, test_identity: dict, fixtures: dict, model=None,
              user_id: str = "u_1120") -> RunResult:
    # Fixtures are trusted test data: {order_id: {"tenant_id": ...}}
    adapters.reset([
        adapters.Order(oid, f["tenant_id"], user_id, Decimal("800")) for oid, f in fixtures.items()
    ])
    order_id = next(iter(fixtures))
    # Scripted "worst case" model: it always tries to refund whatever order is named.
    model = model or ScriptedChatModel(script=[
        ai("", ("issue_refund", {"order_id": order_id, "amount": "800", "reason": "requested"})),
        ai("I could not process that."),
    ])
    ctx = AppContext(user_id=user_id, tenant_id=test_identity["tenant_id"],
                     refund_operation_id="test-op-1")
    out = build_agent(model).invoke({"messages": [("user", request)]}, context=ctx)
    msgs = out["messages"]
    tool_msgs = [m for m in msgs if getattr(m, "type", "") == "tool"]
    denied = any("not found or not accessible" in str(m.content) for m in tool_msgs)
    leaked = sum(1 for m in tool_msgs
                 for f in fixtures.values()
                 if f["tenant_id"] != ctx.tenant_id and f["tenant_id"] in str(m.content))
    refunded = len(adapters.payments.calls) > 0
    return RunResult(
        status="denied" if denied and not refunded else ("completed" if refunded else "no_action"),
        tool_calls=list(iter_tool_names(msgs)),
        cross_tenant_records_returned=leaked,
        final_text=str(msgs[-1].content),
    )
