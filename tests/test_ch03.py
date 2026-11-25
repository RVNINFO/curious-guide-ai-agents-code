import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
from langchain_core.documents import Document
from langgraph.types import Command
from pydantic import ValidationError

from common.fakes import FakeStructuredModel, ScriptedChatModel, TextModel, ai

DATA = Path(__file__).parent.parent / "data"


# ---- 3.1.5 ----
def test_agent_loop_runs_tool_then_answers():
    from ch03.l3_1_5_agent_loop import run_agent
    from common.mini_agent import Call, FakeModel, Response, Tool
    model = FakeModel([Response(tool_calls=[Call("1", "add", {"a": 2, "b": 3})]),
                       Response(content="5")])
    assert run_agent(model, "2+3?", [Tool("add", lambda a, b: a + b)]) == "5"


def test_agent_loop_survives_unknown_tool_bad_args_and_exceptions():
    from ch03.l3_1_5_agent_loop import run_agent
    from common.mini_agent import Call, FakeModel, Response, Tool

    def boom(**_):
        raise RuntimeError("x")

    model = FakeModel([Response(tool_calls=[Call("1", "nope", {}), Call("2", "boom", {})]),
                       Response(content="recovered")])
    assert run_agent(model, "go", [Tool("boom", boom)]) == "recovered"


def test_agent_loop_step_limit():
    from ch03.l3_1_5_agent_loop import run_agent
    from common.mini_agent import Call, FakeModel, Response, Tool
    model = FakeModel([Response(tool_calls=[Call("1", "t", {})])])
    assert run_agent(model, "go", [Tool("t", lambda: 1)], max_steps=3).startswith("I wasn't able")
    assert model.i == 3


# ---- 3.2.3 / 3.3.1 ----
def test_lookup_order():
    from ch03.l3_2_3_order_status_tool import lookup_order
    assert lookup_order.invoke({"order_id": "ORD-1234"})["carrier"] == "BlueDart"
    assert "error" in lookup_order.invoke({"order_id": "ORD-0000"})
    assert "Do NOT use this to find orders" in lookup_order.description


def test_search_orders_schema_constraints_and_json_form_agree():
    from ch03.l3_3_1_tool_schema import SEARCH_ORDERS_JSON, SearchOrdersInput, search_orders
    schema = SearchOrdersInput.model_json_schema()["properties"]
    json_props = SEARCH_ORDERS_JSON["parameters"]["properties"]
    assert schema["limit"]["maximum"] == json_props["limit"]["maximum"] == 20
    assert set(json_props["status"]["enum"]) == {"pending", "shipped", "delivered", "canceled"}
    with pytest.raises(ValidationError):
        SearchOrdersInput(email="a@b.c", limit=50)
    assert len(search_orders.invoke({"email": "ana@example.com", "limit": 1})) == 1


# ---- 3.3.2 ----
def test_triage_structured_output():
    from ch03.l3_3_2_structured_output import TicketTriage, triage
    fake = FakeStructuredModel([{"category": "billing", "urgency": "high", "summary": "double charge",
                                 "suggested_owner": "billing-team", "needs_human": False}])
    r = triage("charged twice", model=fake)
    assert isinstance(r, TicketTriage) and r.category == "billing"


def test_triage_schema_rejects_bad_values():
    from ch03.l3_3_2_structured_output import TicketTriage
    with pytest.raises(ValidationError):
        TicketTriage(category="sales", urgency="low", summary="x", suggested_owner="y", needs_human=False)
    with pytest.raises(ValidationError):
        TicketTriage(category="other", urgency="low", summary="x" * 201, suggested_owner="y", needs_human=False)


# ---- 3.3.4 ----
def _setup_registry():
    from ch03 import l3_3_4_execute_tool_call as m
    m.REGISTRY.clear()
    m.RATE_LIMITER.hits.clear()
    m.BACKOFF_SECONDS = 0  # keep the retry tests fast

    class A(m.BaseModel):
        text: str

    m.REGISTRY["echo"] = m.ToolSpec("echo", lambda text: text.upper(), A)
    m.REGISTRY["admin"] = m.ToolSpec("admin", lambda text: text, A, permission="admin")
    m.REGISTRY["pay"] = m.ToolSpec("pay", lambda text: text, A, requires_approval=True)
    m.REGISTRY["slow"] = m.ToolSpec("slow", lambda text: time.sleep(0.5), A)
    m.REGISTRY["bad"] = m.ToolSpec("bad", lambda text: 1 / 0, A)
    return m, m.Context(m.User("u1", {"read"}))


def test_pipeline_all_seven_stages():
    m, ctx = _setup_registry()
    C = m.ToolCall
    assert m.execute_tool_call(C("echo", {"text": "hi"}), ctx) == "HI"
    assert "Unknown tool" in m.execute_tool_call(C("nope", {}), ctx)["message"]
    assert "Invalid arguments" in m.execute_tool_call(C("echo", {}), ctx)["message"]
    assert "permission" in m.execute_tool_call(C("admin", {"text": "x"}), ctx)["message"]
    assert m.execute_tool_call(C("pay", {"text": "x"}), ctx)["status"] == "PENDING_APPROVAL"
    real = m.run_with_retry
    m.run_with_retry = lambda fn, args, seconds, attempts: real(fn, args, 0.05, attempts)
    try:
        assert m.execute_tool_call(C("slow", {"text": "x"}), ctx)["message"].endswith("timed out.")
    finally:
        m.run_with_retry = real
    r = m.execute_tool_call(C("bad", {"text": "x"}), ctx)
    assert r["message"] == "The tool failed after retries." and r["retryable"] is False


def test_pipeline_rate_limit_runs_before_approval():
    m, ctx = _setup_registry()
    C = m.ToolCall
    for _ in range(5):
        assert m.execute_tool_call(C("pay", {"text": "x"}), ctx)["status"] == "PENDING_APPROVAL"
    # a looping agent cannot flood reviewers: the 6th request is rate limited, not queued
    assert "Rate limit" in m.execute_tool_call(C("pay", {"text": "x"}), ctx)["message"]


def test_pipeline_retries_transient_errors_then_succeeds():
    m, ctx = _setup_registry()
    calls = []

    def flaky(text):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("blip")
        return text.upper()

    m.REGISTRY["flaky"] = m.ToolSpec("flaky", flaky, m.REGISTRY["echo"].input_schema)
    assert m.execute_tool_call(m.ToolCall("flaky", {"text": "ok"}), ctx) == "OK" and len(calls) == 3


def test_pipeline_rate_limit_and_truncation():
    m, ctx = _setup_registry()
    for _ in range(5):
        m.execute_tool_call(m.ToolCall("echo", {"text": "a"}), ctx)
    assert "Rate limit" in m.execute_tool_call(m.ToolCall("echo", {"text": "a"}), ctx)["message"]
    assert m.truncate_for_context("x" * 100000, 10).endswith("[truncated]")


# ---- 3.4.1 / 3.6.1 (agents built with real LangChain middleware, scripted models) ----
def test_summarization_agent_builds_and_answers():
    from ch03.l3_4_1_summarization_agent import build_agent
    model = ScriptedChatModel(script=[ai("", ("lookup_order", {"order_id": "ORD-1234"})),
                                      ai("It has shipped.")])
    agent = build_agent(model=model, summary_model=ScriptedChatModel(script=[ai("summary")]))
    out = agent.invoke({"messages": [("user", "Where is ORD-1234?")]})
    assert out["messages"][-1].content == "It has shipped."
    assert "BlueDart" in str(out["messages"][-2].content)


def test_summarization_triggers_when_threshold_crossed():
    from ch03.l3_4_1_summarization_agent import build_agent
    model = ScriptedChatModel(script=[ai("ok")])
    summ = ScriptedChatModel(script=[ai("SUMMARY-OF-EARLIER-CHAT")])
    agent = build_agent(model=model, summary_model=summ, trigger_tokens=50, keep_messages=2)
    history = [("user", "word " * 60), ("assistant", "ack " * 60), ("user", "more " * 60), ("assistant", "ok"),
               ("user", "last question")]
    agent.invoke({"messages": history})
    assert summ.i >= 1, "summary model should have been called once the token trigger was crossed"


def test_todo_planning_agent_writes_todos():
    from ch03.l3_6_1_todo_planning import build_agent
    todos = [{"content": "search", "status": "in_progress"}, {"content": "draft", "status": "pending"}]
    model = ScriptedChatModel(script=[ai("", ("write_todos", {"todos": todos})), ai("done")])
    out = build_agent(model).invoke({"messages": [("user", "do 4 things")]})
    assert out["todos"][0]["content"] == "search"
    assert out["messages"][-1].content == "done"


# ---- 3.5 RAG ----
def test_rag_offline():
    from ch03 import l3_5_rag as r
    emb = r.make_embeddings()
    vecs = r.embed_examples(emb)
    assert len(vecs) == 3 and len(vecs[0]) == 64
    store = r.build_vectorstore(emb)
    hits = r.filtered_search(store)
    assert hits and all(d.metadata["department"] == "support" and d.metadata["language"] == "en" for d in hits)


def test_rag_rerank_wiring_with_fake_compressor():
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from ch03 import l3_5_rag as r

    class TakeFirstTwo(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            return list(documents)[:2]

    store = r.build_vectorstore(r.make_embeddings())
    retriever = r.build_reranking_retriever(store, compressor=TakeFirstTwo())
    assert len(retriever.invoke("Can I return a damaged item after 30 days?")) == 2


# ---- 3.5.6 Retrieval metrics ----
def test_retrieval_metrics_recall_at_k():
    from ch03.l3_5_6_retrieval_metrics import recall_at_k
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == 1.0
    assert recall_at_k(["a", "b", "c"], {"x"}, k=2) == 0.0


def test_retrieval_metrics_reciprocal_rank():
    from ch03.l3_5_6_retrieval_metrics import reciprocal_rank
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


# ---- 3.6.3 ----
def test_generate_valid_retries_with_error_in_prompt():
    from ch03.l3_6_3_generate_valid import generate_valid
    m = TextModel(['{"answer": "x"}', '{"answer": "Paris", "confidence": 0.9}'])
    msgs = [{"role": "user", "content": "q"}]
    out = generate_valid(m, msgs)
    assert out.answer == "Paris"
    assert any("failed validation" in str(x) for x in m.calls[1])


def test_generate_valid_gives_up_and_escalates():
    from ch03.l3_6_3_generate_valid import generate_valid
    with pytest.raises(RuntimeError, match="escalate"):
        generate_valid(TextModel(["not json"]), [{"role": "user", "content": "q"}], max_attempts=2)


# ---- 3.7.1 ----
def _hitl(decision):
    from ch03 import l3_7_1_human_in_the_loop as h
    h.EXECUTED.clear()
    model = ScriptedChatModel(script=[ai("", ("issue_refund", {"order_id": "ORD-1", "amount": 100.0})),
                                      ai("finished")])
    agent = h.build_agent(model)
    cfg = {"configurable": {"thread_id": f"t-{decision['type']}"}}
    first = agent.invoke({"messages": [("user", "refund")]}, config=cfg)
    assert "__interrupt__" in first and h.EXECUTED == []  # paused BEFORE the side effect
    agent.invoke(Command(resume={"decisions": [decision]}), config=cfg)
    return h.EXECUTED


def test_hitl_approve_executes():
    assert _hitl({"type": "approve"}) == [("issue_refund", "ORD-1", 100.0)]


def test_hitl_reject_does_not_execute():
    assert _hitl({"type": "reject", "message": "no"}) == []


# ---- 3.8 patterns ----
def test_router_routes_and_escalates():
    from ch03 import l3_8_1_router as r
    r.AGENTS.clear()
    r.AGENTS["billing"] = type("B", (), {"invoke": lambda s, req, ses: "billed"})()
    handle = r.make_handler(FakeStructuredModel([{"destination": "billing", "reason": "x"},
                                                 {"destination": "technical", "reason": "no agent"}]))
    assert handle("q", {}) == "billed"
    assert handle("q", {})["escalated"] is True  # no agent registered -> human


def test_reflection_stops_on_approved_and_bounds_rounds():
    from ch03.l3_8_3_reflection import _Str, make_with_reflection
    gen, critic = _Str(["v1", "v2"]), _Str(["fix it", "APPROVED"])
    assert make_with_reflection(gen, critic)("t") == "v2"
    gen, critic = _Str(["v1", "v2", "v3"]), _Str(["bad"])
    make_with_reflection(gen, critic)("t", max_rounds=2)
    assert len(critic.prompts) == 2 and len(gen.prompts) == 3


def test_reviewer_paths():
    from ch03.l3_8_5_reviewer import Action, Approved, Escalate, Rejected, _FixedReviewer, make_review
    rev = _FixedReviewer("approve")
    review = make_review(rev)
    assert isinstance(review(Action("issue_refund", {"amount": 5}), {}), Approved) and rev.calls == 1
    assert isinstance(review(Action("issue_refund", {"amount": -5}), {}), Rejected) and rev.calls == 1  # model skipped
    assert isinstance(review(Action("delete_account", {}), {}), Escalate)
    assert isinstance(make_review(_FixedReviewer("uncertain"))(Action("issue_refund", {"amount": 5}), {}), Escalate)
    assert isinstance(make_review(_FixedReviewer("reject"))(Action("issue_refund", {"amount": 5}), {}), Rejected)


# ---- 3.11.1 MCP ----
def test_mcp_server_lists_and_calls_tool():
    from ch03.l3_11_1_mcp_server import mcp

    async def go():
        tools = await mcp.list_tools()
        res = await mcp.call_tool("translate_text", {"text": "Hello", "target_language": "Japanese"})
        return tools, res

    tools, res = asyncio.run(go())
    t = tools[0]
    assert t.name == "translate_text" and "Japanese" in t.description
    assert set(t.parameters["required"]) == {"text", "target_language"}  # matches the book's tools/list
    assert "こんにちは" in str(res)
    book = json.loads((DATA / "mcp_tools_list.jsonrpc.json").read_text(encoding="utf-8"))
    assert book["response"]["result"]["tools"][0]["name"] == t.name


def test_mcp_server_resource_and_prompt():
    from ch03.l3_11_1_mcp_server import mcp

    async def go():
        resources = await mcp.list_resources()
        content = await mcp.read_resource("translation://languages")
        prompts = await mcp.list_prompts()
        got = await mcp.render_prompt("translate_formally", {"text": "hello", "target_language": "Japanese"})
        return resources, content, prompts, got

    resources, content, prompts, got = asyncio.run(go())
    res_wire = json.loads((DATA / "mcp_resources.jsonrpc.json").read_text(encoding="utf-8"))
    prompt_wire = json.loads((DATA / "mcp_prompts.jsonrpc.json").read_text(encoding="utf-8"))
    assert str(resources[0].uri) == res_wire["list"]["response"]["result"]["resources"][0]["uri"]
    assert "Japanese, French, Spanish" in str(content)
    assert prompts[0].name == prompt_wire["list"]["response"]["result"]["prompts"][0]["name"]
    expected = prompt_wire["get"]["response"]["result"]["messages"][0]["content"]["text"]
    assert got.messages[0].content.text == expected


# ---- 3.12.3 A2A ----
def test_a2a_client_async_task_flow():
    from ch03.l3_12_3_a2a_client import ask_agent
    card = json.loads((DATA / "agent_card.json").read_text(encoding="utf-8"))
    polls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("agent-card.json"):
            return httpx.Response(200, json=card)
        body = json.loads(req.content)
        assert req.headers["authorization"] == "Bearer tok"
        assert req.headers["a2a-version"] == "1.0"
        if body["method"] == "SendMessage":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": "req-1",
                                              "result": {"task": {"id": "task-1", "status": {"state": "TASK_STATE_SUBMITTED"}}}})
        polls["n"] += 1
        state = "TASK_STATE_COMPLETED" if polls["n"] >= 2 else "TASK_STATE_WORKING"
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "req-2",
                                          "result": {"id": "task-1", "status": {"state": state},
                                                     "artifacts": [{"name": "stock", "qty": 600}]}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        out = ask_agent(c, "https://supplier.example.com", "tok", "500 units?", poll_seconds=0)
    assert out == [{"name": "stock", "qty": 600}] and polls["n"] == 2


def test_a2a_client_direct_message_flow():
    from ch03.l3_12_3_a2a_client import ask_agent
    card = json.loads((DATA / "agent_card.json").read_text(encoding="utf-8"))

    def handler(req):
        if req.url.path.endswith("agent-card.json"):
            return httpx.Response(200, json=card)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "req-1",
                                          "result": {"message": {"parts": [{"text": "yes"}]}}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        assert ask_agent(c, "https://supplier.example.com", "t", "q", 0)[0]["text"] == "yes"


def test_a2a_client_paused_flow():
    from ch03.l3_12_3_a2a_client import TaskPaused, ask_agent
    card = json.loads((DATA / "agent_card.json").read_text(encoding="utf-8"))

    def handler(req):
        if req.url.path.endswith("agent-card.json"):
            return httpx.Response(200, json=card)
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": "req-1",
            "result": {"task": {
                "id": "task-paused",
                "status": {
                    "state": "TASK_STATE_INPUT_REQUIRED",
                    "message": {"parts": [{"text": "Which warehouse?"}]}
                }
            }}
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(TaskPaused) as exc:
            ask_agent(c, "https://supplier.example.com", "t", "q", 0)
        assert exc.value.state == "TASK_STATE_INPUT_REQUIRED"
        assert "Which warehouse?" in exc.value.question
        assert "TASK_STATE_INPUT_REQUIRED. Which warehouse?" in str(exc.value)  # space between f-strings


def test_a2a_client_unique_message_ids_and_http_errors():
    from ch03.l3_12_3_a2a_client import ask_agent
    card = json.loads((DATA / "agent_card.json").read_text(encoding="utf-8"))
    ids, fail = [], {"on": False}

    def handler(req):
        if req.url.path.endswith("agent-card.json"):
            return httpx.Response(200, json=card)
        if fail["on"]:
            return httpx.Response(503, text="unavailable")
        ids.append(json.loads(req.content)["params"]["message"]["messageId"])
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "req-1",
                                          "result": {"message": {"parts": [{"text": "yes"}]}}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        ask_agent(c, "https://s.example.com", "t", "q", 0)
        ask_agent(c, "https://s.example.com", "t", "q", 0)
        assert len(set(ids)) == 2
        fail["on"] = True
        with pytest.raises(httpx.HTTPStatusError):
            ask_agent(c, "https://s.example.com", "t", "q", 0)


def test_a2a_client_timeout():
    from ch03.l3_12_3_a2a_client import ask_agent
    card = json.loads((DATA / "agent_card.json").read_text(encoding="utf-8"))

    def handler(req):
        if req.url.path.endswith("agent-card.json"):
            return httpx.Response(200, json=card)
        body = json.loads(req.content)
        if body["method"] == "SendMessage":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": "req-1",
                                              "result": {"task": {"id": "task-1", "status": {"state": "TASK_STATE_WORKING"}}}})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "req-2",
                                          "result": {"id": "task-1", "status": {"state": "TASK_STATE_WORKING"}}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(TimeoutError):
            ask_agent(c, "https://supplier.example.com", "t", "q", poll_seconds=0.01, timeout_seconds=0.02)

