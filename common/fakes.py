"""Offline test doubles so every listing can run without an API key.

ScriptedChatModel plays back a fixed list of AIMessages (including tool calls),
so agent behaviour is deterministic. Live runs simply pass a real model string
instead (see each listing's `build_agent` / `main`).
"""
from __future__ import annotations

from typing import Any, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def ai(content: str = "", *calls: tuple[str, dict]) -> AIMessage:
    """Build an AIMessage; each call is (tool_name, args)."""
    return AIMessage(
        content=content,
        tool_calls=[
            {"name": n, "args": a, "id": f"call_{i}_{n}", "type": "tool_call"}
            for i, (n, a) in enumerate(calls)
        ],
    )


class ScriptedChatModel(BaseChatModel):
    """Returns scripted messages in order; repeats the last one when exhausted."""

    script: list[Any]
    i: int = 0
    seen: list[Any] = []  # message lists the model was called with

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # tools are ignored; the script decides
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw) -> ChatResult:
        self.seen.append(list(messages))
        msg = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        if callable(msg):  # allow script entries that inspect the messages
            msg = msg(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeStructuredModel:
    """Stands in for `init_chat_model(...)` when a listing uses with_structured_output."""

    def __init__(self, payloads: list[dict]):
        self.payloads, self.i = payloads, 0

    def with_structured_output(self, schema):
        outer = self

        class _Runner:
            def invoke(self, _input):
                p = outer.payloads[min(outer.i, len(outer.payloads) - 1)]
                outer.i += 1
                return schema(**p)

        return _Runner()


class TextModel:
    """Minimal `.invoke(x)` model returning objects with `.content` (used by pseudocode listings)."""

    def __init__(self, replies: list[str]):
        self.replies, self.i, self.calls = replies, 0, []

    def invoke(self, x):
        self.calls.append(x)
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return type("Msg", (), {"content": r, "__str__": lambda s: r})()


def iter_tool_names(messages) -> Iterator[str]:
    for m in messages:
        for c in getattr(m, "tool_calls", None) or []:
            yield c["name"]
