"""Listing 3.2.3 - A well-described tool (LangChain @tool).

The `db` client is the in-memory stand-in from common/fake_db.py.
Run:  python -m ch03.l3_2_3_order_status_tool
"""
from langchain_core.tools import tool

from common.fake_db import db


@tool
def get_order_status(order_id: str) -> dict:
    """Look up the current status of a single order by its ID.

    Use this when the customer refers to a specific order and you
    already have its ID (format: ORD-1234)

    Do NOT use this to find orders by customer name or email --
    use search_orders for that.

    Returns status, carrier, tracking number and estimated
    delivery date, or an error message if the order is not found.
    """
    order = db.orders.find_one({"id": order_id})
    if order is None:
        return {"error": f"No order found with ID {order_id}."}
    return {
        "status": order["status"],
        "carrier": order["carrier"],
        "tracking": order["tracking_number"],
        "eta": order["estimated_delivery"],
    }


if __name__ == "__main__":
    print(get_order_status.invoke({"order_id": "ORD-1234"}))
    print(get_order_status.invoke({"order_id": "ORD-0000"}))
