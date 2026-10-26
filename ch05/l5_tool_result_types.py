"""Listing 5.x - A typed tool-result contract: every tool returns success or failure, never prose.

Run:  python -m ch05.l5_tool_result_types
"""
from typing import Any, Literal, TypedDict


class ToolSuccess(TypedDict):
    success: Literal[True]
    data: Any


class ToolFailure(TypedDict):
    success: Literal[False]
    error_code: str
    message: str
    retryable: bool


ToolResult = ToolSuccess | ToolFailure


def ok(data: Any) -> ToolSuccess:
    return {"success": True, "data": data}


def fail(error_code: str, message: str, retryable: bool = False) -> ToolFailure:
    return {"success": False, "error_code": error_code, "message": message, "retryable": retryable}


def should_retry(result: ToolResult) -> bool:
    return (not result["success"]) and result["retryable"]


if __name__ == "__main__":
    print(ok({"reference": "RF-29381"}), fail("NOT_FOUND", "No such order"))
