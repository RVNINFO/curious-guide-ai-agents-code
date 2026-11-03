"""Listing 3.3.1 - A tool definition: raw JSON Schema form, then Pydantic form.

Run:  python -m ch03.l3_3_1_tool_schema
"""
import json
from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.fake_db import db

# The JSON Schema form (what every framework ultimately produces) lives in data/.
SEARCH_ORDERS_JSON = json.loads(
    (Path(__file__).parent.parent / "data" / "search_orders.tool.json").read_text(encoding="utf-8")
)


# The Pydantic form: write Python, let the framework generate the schema.
class SearchOrdersInput(BaseModel):
    email: str = Field(description="The customer's email address.")
    status: Optional[Literal["pending", "shipped",
                             "delivered", "canceled"]] = Field(
        default=None,
        description="Optional filter. Omit to return all statuses.")
    limit: int = Field(default=10, ge=1, le=20,
                       description="How many orders to return.")


@tool(args_schema=SearchOrdersInput)
def search_orders(email: str, status=None, limit=10) -> list[dict]:
    """Search for a customer's orders by email address.

    Returns up to 20 recent orders, newest first. Use this when the
    customer has not given a specific order ID. If you already have
    an order ID, use get_order_status instead.
    """
    query = {"email": email}
    if status:
        query["status"] = status
    orders = (db.orders.find(query)
              .sort("created_at", -1)
              .limit(limit))
    return [{"id": o["id"], "status": o["status"],
             "created": o["created_at"]} for o in orders]


if __name__ == "__main__":
    print(json.dumps(search_orders.args_schema.model_json_schema(), indent=2))
    print(search_orders.invoke({"email": "ana@example.com", "status": "shipped"}))
