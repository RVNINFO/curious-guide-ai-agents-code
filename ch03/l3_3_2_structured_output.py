"""Listing 3.3.2 - Structured output with a Pydantic schema.

Offline: pass any object with `.with_structured_output` (see common/fakes.py).
Live:    ANTHROPIC_API_KEY=... python -m ch03.l3_3_2_structured_output
"""
from typing import Literal

from pydantic import BaseModel, Field


class TicketTriage(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high", "critical"]
    summary: str = Field(max_length=200)
    suggested_owner: str
    needs_human: bool = Field(
        description="True if this requires human judgment.")


def triage(ticket_text: str, model=None) -> TicketTriage:
    if model is None:  # live mode: any provider:model string works
        from langchain.chat_models import init_chat_model
        model = init_chat_model("anthropic:claude-sonnet-5-5")
    structured_model = model.with_structured_output(TicketTriage)
    return structured_model.invoke(ticket_text)


if __name__ == "__main__":
    result = triage("I was charged twice for my October invoice and need it fixed today.")
    print(result.category, result.urgency)  # a real object, not a string
