"""A scripted stand-in for the real model, so the capstone evals run offline and deterministically.

It is *reactive*, like a sensible model: it looks up the order named in the customer's message, reads the
tool result, and only then decides what to do next (refund / large refund / give up). It encodes what a
correct model should do, so these tests exercise the harness and guardrails (ownership, limits, approval,
idempotency, eval scoring) - not any real model's judgment. Use `--live` for that.
"""
import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from common.fakes import ai


def _as_json(text):
    """Parse a tool result. After a human *edits* a call, LangChain's approval middleware prepends a
    plain-text note ending in "Tool response:" to the tool message, so parse what follows it.
    Plain text with no JSON (a rejection) -> None."""
    try:
        return json.loads(text.split("Tool response:\n")[-1])
    except (TypeError, ValueError):
        return None


def _calls(*calls):
    return ai("", *calls)


class ScriptedSupportModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "scripted-support"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self, messages) -> AIMessage:
        user = next(m.content for m in messages if m.type == "human")
        order_id = re.search(r"ORD-\d+", user).group()
        tools = [m for m in messages if m.type == "tool"]
        if not tools:                                   # step 1: investigate
            return _calls(("lookup_order", {"order_id": order_id}),
                          ("search_policy", {"query": "damaged refund approval"}))
        by_name = {m.name: m for m in tools}
        if "issue_refund" in by_name or "issue_large_refund" in by_name:      # step 3: report
            last = by_name.get("issue_large_refund") or by_name["issue_refund"]
            res = _as_json(last.content)
            if res is None:                              # e.g. a human rejected the approval: plain text, not JSON
                return ai("A supervisor did not approve this refund, so nothing was paid.")
            return ai(f"Your refund has been processed. Reference {res['data']['reference']}."
                      if res.get("success") else f"I could not complete the refund: {res.get('message')}")
        order = json.loads(by_name["lookup_order"].content)
        if not order.get("success"):                    # not found / not yours: stop and say so
            return ai("I couldn't find that order on your account. Could you check the order ID?")
        amount = order["data"]["amount"]                # string like "800.00"
        tool = "issue_large_refund" if float(amount) > 5000 else "issue_refund"
        return _calls((tool, {"order_id": order_id, "amount": amount}))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])
