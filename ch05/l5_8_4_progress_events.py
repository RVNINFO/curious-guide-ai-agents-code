"""Listing 5.8.4 - Progress as an event stream (FastAPI + Server-Sent Events), using the Section 5.11.1 agent.

The book's snippet assumes `build_agent` and `AppContext` from 5.11.1; here they are imported.
`create_app(agent)` wraps the printed code so tests can inject a scripted agent.
Run a dev server:  uvicorn ch05.l5_8_4_progress_events:app   (needs ANTHROPIC_API_KEY; live model)
"""
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ch05.l5_11_1_capstone import AppContext, build_agent

# Fixed labels.
# Order IDs, amounts and prompts never reach the browser.
PROGRESS = {
    "lookup_order": "Checking your order",
    "search_policy": "Checking the return policy",
    "issue_refund": "Issuing the refund",
    "issue_large_refund": "Preparing a refund for approval",
}


def to_event(chunk: dict):
    """Turn one agent update into a safe progress event,
    or None to stay quiet."""

    if "__interrupt__" in chunk:
        return {"type": "awaiting_approval",
                "text": "Waiting for manager approval"}
    for update in chunk.values():
        for msg in (update or {}).get("messages", []):
            if msg.type != "ai":
                continue
            if msg.tool_calls:
                name = msg.tool_calls[0]["name"]
                return {"type": "step",
                        "text": PROGRESS.get(
                            name, "Working on it")}
            return {"type": "final", "text": msg.text}
    return None


def create_app(agent) -> FastAPI:
    app = FastAPI()

    @app.get("/runs/{thread_id}/events")
    def events(thread_id: str, message: str):
        def stream():
            config = {"configurable": {"thread_id": thread_id}}
            for n, chunk in enumerate(agent.stream(
                    {"messages": [("user", message)]},
                    config=config,
                    # from your login system
                    context=AppContext(user_id="u_1120"),
                    stream_mode="updates")):
                event = to_event(chunk)
                if event:
                    # In production, also save the event to the
                    # task record so a client that reconnects
                    # with Last-Event-ID can replay it.
                    yield f"id: {n}\ndata: {json.dumps(event)}\n\n"
        return StreamingResponse(stream(),
                                 media_type="text/event-stream")

    return app


def __getattr__(name):  # `uvicorn ch05.l5_8_4_progress_events:app` builds the live app lazily
    if name == "app":
        return create_app(build_agent())
    raise AttributeError(name)
