"""Listing 3.3.4 - The tool-execution pipeline (7 checks between request and result).

The book prints this as pseudocode. The helpers it assumes (REGISTRY, error, RATE_LIMITER,
run_with_timeout, request_approval, truncate_for_context, log) are implemented minimally here.
Run:  python -m ch03.l3_3_4_execute_tool_call
"""
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

log = logging.getLogger("tools")


# --- helpers the book assumes ---------------------------------------------
@dataclass
class ToolSpec:
    name: str
    fn: Callable
    input_schema: type[BaseModel]
    permission: str = "read"
    requires_approval: bool = False


@dataclass
class User:
    id: str
    permissions: set = field(default_factory=set)

    def can(self, permission: str) -> bool:
        return permission in self.permissions


@dataclass
class Context:
    user: User


@dataclass
class ToolCall:
    name: str
    arguments: dict


class RateLimiter:
    def __init__(self, max_calls: int, per_seconds: float):
        self.max_calls, self.per, self.hits = max_calls, per_seconds, defaultdict(list)

    def allow(self, user_id: str, tool_name: str) -> bool:
        now, key = time.monotonic(), (user_id, tool_name)
        self.hits[key] = [t for t in self.hits[key] if now - t < self.per]
        if len(self.hits[key]) >= self.max_calls:
            return False
        self.hits[key].append(now)
        return True


REGISTRY: dict[str, ToolSpec] = {}
RATE_LIMITER = RateLimiter(max_calls=5, per_seconds=60)


def error(message: str, retryable: bool = False) -> dict:
    return {"success": False, "message": message, "retryable": retryable}


def request_approval(call, context) -> dict:
    return {"success": False, "status": "PENDING_APPROVAL", "tool": call.name}


def run_with_timeout(fn, args: BaseModel, seconds: float):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, **args.model_dump())
        try:
            return future.result(timeout=seconds)
        except FutureTimeout as exc:
            raise TimeoutError from exc


def truncate_for_context(result: Any, max_tokens: int = 2000) -> Any:
    text = str(result)
    limit = max_tokens * 4  # rough chars-per-token estimate
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


# --- the listing -----------------------------------------------------------
def execute_tool_call(call, context, timeout_seconds: float = 30):
    # 1. Does this tool exist?
    tool = REGISTRY.get(call.name)
    if tool is None:
        return error(f"Unknown tool: {call.name}")

    # 2. Are the arguments valid?
    try:
        args = tool.input_schema(**call.arguments)
    except ValidationError as e:
        return error(f"Invalid arguments: {e}")

    # 3. Is this user allowed to do this?
    if not context.user.can(tool.permission):
        return error("You do not have permission for this action.")

    # 4. Does it need approval?
    if tool.requires_approval:
        return request_approval(call, context)

    # 5. Rate limit
    if not RATE_LIMITER.allow(context.user.id, tool.name):
        return error("Rate limit reached. Try again shortly.")

    # 6. Run it, with a timeout
    try:
        result = run_with_timeout(tool.fn, args, seconds=timeout_seconds)
    except TimeoutError:
        return error("The operation timed out.")
    except Exception:
        log.exception("tool_failed: %s", call.name)
        return error("ERR_INTERNAL_FAILURE", retryable=True)

    # 7. Trim before it enters the context
    return truncate_for_context(result, max_tokens=2000)


class _EchoArgs(BaseModel):
    text: str


if __name__ == "__main__":
    REGISTRY["echo"] = ToolSpec("echo", lambda text: text.upper(), _EchoArgs)
    ctx = Context(User("u_1", {"read"}))
    print(execute_tool_call(ToolCall("echo", {"text": "hello"}), ctx))
    print(execute_tool_call(ToolCall("nope", {}), ctx))
