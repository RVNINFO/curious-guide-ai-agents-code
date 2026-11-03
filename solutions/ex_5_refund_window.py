"""Reference solution for the Part 5 exercise (refund window enforced in the tool)."""
from langchain_core.tools import tool

from exercises.ex_5_refund_window_starter import APPROVAL_LIMIT, ORDERS

WINDOW_DAYS = 30


@tool
def issue_refund(order_id: str, amount: float) -> dict:
    """Issue a refund of up to ₹5,000 for an order delivered within the last 30 days."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"success": False,
                "error_code": "NOT_FOUND",
                "retryable": False,
                "message": f"No order found with ID {order_id}."}
    if order["delivered_days_ago"] > WINDOW_DAYS:
        return {"success": False,
                "error_code": "OUTSIDE_WINDOW",
                "retryable": False,
                "message": "Delivered more than 30 days ago."}

    if amount > APPROVAL_LIMIT:
        return {"success": False, "error_code": "NEEDS_APPROVAL",
                "message": "Amount exceeds the automatic limit.", "retryable": False}

    return {"success": True, "data": {"order_id": order_id, "amount": amount}}
