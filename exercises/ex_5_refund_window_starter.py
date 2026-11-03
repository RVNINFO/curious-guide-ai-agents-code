"""Exercise starter (end of Part 5): enforce the 30-day window inside the tool, not the prompt."""
from decimal import Decimal

from langchain_core.tools import tool

APPROVAL_LIMIT = Decimal("5000")  # INR (₹)
ORDERS = {
    "ORD-4821": {"owner": "u_1120", "delivered_days_ago": 3,
                 "amount": Decimal("800")},
    "ORD-4822": {"owner": "u_1120", "delivered_days_ago": 5,
                 "amount": Decimal("25000")},
    "ORD-4823": {"owner": "u_1120", "delivered_days_ago": 45,
                 "amount": Decimal("1200")},
}


@tool
def issue_refund(order_id: str, amount: float) -> dict:
    """Issue a refund of up to ₹5,000 for an order delivered within the last 30 days."""
    # TODO: look up the order.
    # Refuse (success False, error_code "OUTSIDE_WINDOW", retryable False)
    # if it was delivered more than 30 days ago, or if the
    # order does not exist (error_code "NOT_FOUND").
    if amount > APPROVAL_LIMIT:
        return {
            "success": False,
            "error_code": "NEEDS_APPROVAL",
            "message": "Amount exceeds the automatic limit.",
            "retryable": False,
        }

    return {"success": True, "data": {"order_id": order_id, "amount": amount}}
