"""Listing 3.8.1 - Router pattern. A cheap model classifies; specialist agents handle.

Run:  python -m ch03.l3_8_1_router
"""
from typing import Literal

from pydantic import BaseModel


class Route(BaseModel):
    destination: Literal["billing", "technical", "returns", "human"]
    reason: str


ROUTER_PROMPT = "Route this customer request to billing, technical, returns, or human.\n\n"
AGENTS: dict = {}  # destination -> object with .invoke(request, session)


def escalate_to_human(request, reason):
    return {"escalated": True, "reason": reason, "request": request}


def make_handler(small_model):
    router = small_model.with_structured_output(Route)  # cheap model

    def handle(request, session):
        route = router.invoke(ROUTER_PROMPT + request)
        agent = AGENTS.get(route.destination)
        if agent is None:
            return escalate_to_human(request, route.reason)
        return agent.invoke(request, session)

    return handle


if __name__ == "__main__":
    from common.fakes import FakeStructuredModel

    class Billing:
        def invoke(self, request, session):
            return f"billing agent handled: {request}"

    AGENTS["billing"] = Billing()
    handle = make_handler(FakeStructuredModel([
        {"destination": "billing", "reason": "invoice question"},
        {"destination": "human", "reason": "angry customer"},
    ]))
    print(handle("Why was I charged twice?", session={}))
    print(handle("This is unacceptable!", session={}))
