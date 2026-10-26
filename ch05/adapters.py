"""In-memory versions of the 'application-owned adapters' that Listing 5.x says you must supply:
`db.orders.get_for_tenant`, `db.refunds.total_for_order`, and `payments.refund`.
Every query takes the tenant explicitly - there is deliberately no global lookup.
"""
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class Order:
    id: str
    tenant_id: str
    user_id: str
    total: Decimal


class _Orders:
    def __init__(self):
        self.rows: list[Order] = []

    def get_for_tenant(self, tenant_id: str, order_id: str):
        return next((o for o in self.rows if o.tenant_id == tenant_id and o.id == order_id), None)


class _Refunds:
    def __init__(self):
        self.rows: list[dict] = []

    def total_for_order(self, tenant_id: str, order_id: str) -> Decimal:
        return sum((r["amount"] for r in self.rows
                    if r["tenant_id"] == tenant_id and r["order_id"] == order_id), Decimal("0"))


class _DB:
    def __init__(self):
        self.orders, self.refunds = _Orders(), _Refunds()


class _Payments:
    def __init__(self):
        self.calls: list[dict] = []

    def refund(self, tenant_id, order_id, amount, reason, idempotency_key) -> dict:
        if any(c["idempotency_key"] == idempotency_key for c in self.calls):
            return {"success": True, "data": {"duplicate": True}}  # replay is a no-op
        self.calls.append(dict(tenant_id=tenant_id, order_id=order_id, amount=amount,
                               reason=reason, idempotency_key=idempotency_key))
        db.refunds.rows.append({"tenant_id": tenant_id, "order_id": order_id, "amount": amount})
        return {"success": True, "data": {"reference": "RF-29381", "amount": str(amount)}}


db = _DB()
payments = _Payments()


def reset(orders: list[Order] | None = None) -> None:
    db.orders.rows = list(orders or [])
    db.refunds.rows = []
    payments.calls = []
