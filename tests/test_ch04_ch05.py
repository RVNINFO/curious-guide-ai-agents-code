import json
from decimal import Decimal
from pathlib import Path

import pytest
from langgraph.types import Command

from common.fakes import ScriptedChatModel, TextModel, ai

DATA = Path(__file__).parent.parent / "data"


# ---- Part 4 ----
def test_judge_parses_json_scores():
    from ch04.l4_2_3_llm_judge import JUDGE_PROMPT, judge
    m = TextModel(['{"grounded": 5, "complete": 4, "correct": 5, "tone": 4, "notes": "ok"}'])
    out = judge(m, "q", "ctx", "a")
    assert out["grounded"] == 5
    assert "CUSTOMER ASKED:\nq" in m.calls[0] and '"grounded": n' in m.calls[0]
    assert "{question}" in JUDGE_PROMPT


def test_process_scoring_checks_tools_and_answer():
    from ch04.l4_2_3_llm_judge import load_case, score_process
    case = load_case()
    good = {"get_order_status", "start_return"}
    assert score_process(case, good, "You can return it within 30 days.")
    assert not score_process(case, good | {"issue_refund"}, "return within 30 days")
    assert not score_process(case, {"get_order_status"}, "return within 30 days")
    assert not score_process(case, good, "refund approved, return in 30 days")


# ---- 4.2.9 Pass rates ----
@pytest.mark.parametrize("outcomes,expected_rate,expected_pass_at_n,expected_all_n", [
    ([True, True, True, True, True], 1.0, True, True),
    ([True, True, False, True, True], 0.8, True, False),
    ([False, True, False, False, True], 0.4, True, False),
    ([False, False, False], 0.0, False, False),
    ([True], 1.0, True, True),
    ([False], 0.0, False, False),
])
def test_pass_rates_scenarios(outcomes, expected_rate, expected_pass_at_n, expected_all_n):
    from ch04.l4_2_9_pass_rates import score_case
    r = score_case(outcomes)
    assert r["pass_rate"] == expected_rate
    assert r["pass_at_n"] is expected_pass_at_n
    assert r["all_n"] is expected_all_n


def test_pass_rates_scoring():
    from ch04.l4_2_9_pass_rates import score_case
    r1 = score_case([True, True, True])
    assert r1["pass_rate"] == 1.0 and r1["all_n"] and r1["pass_at_n"]
    r2 = score_case([True, False, True])
    assert round(r2["pass_rate"], 2) == 0.67 and not r2["all_n"] and r2["pass_at_n"]
    r3 = score_case([False, False])
    assert r3["pass_rate"] == 0.0 and not r3["all_n"] and not r3["pass_at_n"]


# ---- Part 5: tenant-scoped refund ----
def test_cross_tenant_refund_is_denied():
    from ch05.harness import run_agent
    result = run_agent(
        request="Check order ORD-4202 for a refund",
        test_identity={"tenant_id": "tenant-a"},
        fixtures={"ORD-4202": {"tenant_id": "tenant-b"}},
    )
    assert result.status == "denied"
    assert result.cross_tenant_records_returned == 0
    from ch05 import adapters
    assert adapters.payments.calls == []
    assert result.tool_calls_named("issue_refund") == 1


def test_same_tenant_refund_succeeds_and_is_idempotent():
    from ch05 import adapters
    from ch05.harness import run_agent
    r = run_agent("Refund ORD-1", {"tenant_id": "tenant-a"}, {"ORD-1": {"tenant_id": "tenant-a"}})
    assert r.status == "completed" and len(adapters.payments.calls) == 1
    key = adapters.payments.calls[0]["idempotency_key"]
    assert key == "refund:tenant-a:test-op-1:ORD-1:800.00"
    assert adapters.payments.refund("tenant-a", "ORD-1", Decimal("1"), "x", key)["data"]["duplicate"]
    assert len(adapters.payments.calls) == 1


def _invoke_refund(order_tenant, user_id, amount, order_total="800", ctx_user="u_1120"):
    from ch05 import adapters
    from ch05.l5_tenant_scoped_refund import AppContext, issue_refund
    adapters.reset([adapters.Order("O1", order_tenant, user_id, Decimal(order_total))])
    from langchain.tools import ToolRuntime  # noqa: F401
    m = ScriptedChatModel(script=[ai("", ("issue_refund", {"order_id": "O1", "amount": amount, "reason": "r"})),
                                  ai("x")])
    from ch05.l5_tenant_scoped_refund import build_agent
    out = build_agent(m).invoke({"messages": [("user", "go")]},
                                context=AppContext(ctx_user, "tenant-a", "op-1"))
    return out["messages"][-2].content


def test_refund_rules():
    assert "not found or not accessible" in _invoke_refund("tenant-b", "u_1120", "100")
    assert "does not belong" in _invoke_refund("tenant-a", "u_other", "100")
    assert "Only 800 remains" in _invoke_refund("tenant-a", "u_1120", "900")
    assert "supervisor" in _invoke_refund("tenant-a", "u_1120", "6000", order_total="9000")
    assert "RF-29381" in _invoke_refund("tenant-a", "u_1120", "100")


def test_refund_args_reject_nonpositive_and_fractional_paise():
    from pydantic import ValidationError
    from ch05.l5_tenant_scoped_refund import RefundArgs
    for bad in ("0", "-5", "10.123"):
        with pytest.raises(ValidationError):
            RefundArgs(order_id="o", amount=bad, reason="r")


def test_tool_result_types_and_retry_rule():
    from ch05.l5_tool_result_types import fail, ok, should_retry
    assert not should_retry(ok(1))
    assert should_retry(fail("TIMEOUT", "slow", retryable=True))
    assert not should_retry(fail("NOT_FOUND", "no"))


@pytest.mark.parametrize("status,retryable,expected_retry", [
    ("TIMEOUT", True, True),
    ("RATE_LIMIT", True, True),
    ("SERVICE_UNAVAILABLE", True, True),
    ("NOT_FOUND", False, False),
    ("INVALID_ARGUMENTS", False, False),
    ("PERMISSION_DENIED", False, False),
])
def test_tool_result_types_retry_matrix(status, retryable, expected_retry):
    from ch05.l5_tool_result_types import fail, should_retry
    assert should_retry(fail(status, "msg", retryable=retryable)) is expected_retry


# ---- 5.8.4 Progress events ----
def test_progress_events_to_event():
    from ch05.l5_8_4_progress_events import to_event
    from langchain_core.messages import AIMessage
    ai_call = AIMessage(content="", tool_calls=[{"name": "lookup_order", "args": {}, "id": "c1"}])
    assert to_event({"node": {"messages": [ai_call]}}) == {"type": "step", "text": "Checking your order"}

    assert to_event({"__interrupt__": ()}) == {"type": "awaiting_approval", "text": "Waiting for manager approval"}

    ai_final = AIMessage(content="Refund done.")
    assert to_event({"node": {"messages": [ai_final]}}) == {"type": "final", "text": "Refund done."}


def test_progress_events_fastapi_sse_stream():
    from starlette.testclient import TestClient
    from ch05.l5_8_4_progress_events import create_app
    from ch05.scripted import ScriptedSupportModel
    from ch05.l5_11_1_capstone import build_agent
    app = create_app(build_agent(ScriptedSupportModel()))
    client = TestClient(app)
    resp = client.get("/runs/test-sse/events?message=ORD-4821 arrived damaged. Refund please.")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    lines = resp.text.split("\n\n")
    events = [json.loads(l.split("data: ")[1]) for l in lines if "data: " in l]
    assert any(e["type"] == "step" for e in events)
    assert any(e["type"] == "final" for e in events)


# ---- 5.9.4 SQL validator ----
@pytest.mark.parametrize("query,expected_ok,expected_snippet", [
    ("SELECT * FROM invoices", True, "LIMIT 200"),
    ("SELECT * FROM invoices LIMIT 10", True, "LIMIT 10"),
    ("SELECT * FROM invoices LIMIT 1000000", True, "LIMIT 200"),
    ("SELECT id FROM purchase_orders", True, "LIMIT 200"),
    ("SELECT * FROM goods_receipts", True, "LIMIT 200"),
    ("SELECT count(*) FROM invoices", True, "LIMIT 200"),
    ("WITH inv AS (SELECT * FROM invoices) SELECT * FROM inv", True, "LIMIT 200"),
    ("DELETE FROM invoices WHERE id = 1", False, "SELECT only"),
    ("INSERT INTO invoices VALUES (1)", False, "SELECT only"),
    ("UPDATE invoices SET amount = 0", False, "SELECT only"),
    ("SELECT * FROM users", False, "table not allowed"),
    ("SELECT * FROM secrets", False, "table not allowed"),
    ("SELECT * FROM other_schema.invoices", False, "schema qualification not allowed"),
    ("SELECT pg_sleep(5) FROM invoices", False, "function not allowed"),
    ("SELECT pg_read_file('x') FROM invoices", False, "function not allowed"),
    ("SELECT FROM WHERE", False, "unparseable"),
    ("SELECT 1; SELECT 2", False, "exactly one statement"),
])
def test_validate_query_cases(query, expected_ok, expected_snippet):
    from ch05.l5_9_4_validate_query import validate
    ok, res = validate(query)
    assert ok is expected_ok
    assert expected_snippet.lower() in res.lower()


def test_validate_query_sqlglot():
    from ch05.l5_9_4_validate_query import validate
    ok, q = validate("SELECT * FROM invoices")
    assert ok and "LIMIT 200" in q
    ok, q = validate("SELECT * FROM invoices LIMIT 500")
    assert ok and "LIMIT 200" in q
    ok, err = validate("DROP TABLE invoices")
    assert not ok and err == "SELECT only"
    ok, err = validate("SELECT * FROM users")
    assert not ok and "table not allowed" in err
    ok, err = validate("SELECT * FROM public.invoices")
    assert not ok and "schema qualification not allowed" in err
    ok, err = validate("SELECT pg_sleep(1) FROM invoices")
    assert not ok and "function not allowed" in err
    ok, err = validate("SELECT 1; SELECT 2")
    assert not ok and err == "exactly one statement"


# ---- Part 5: capstone 5.11.1 ----
def _rt(user_id="u_1120"):
    from types import SimpleNamespace
    from ch05.l5_11_1_capstone import AppContext
    return SimpleNamespace(context=AppContext(user_id=user_id))


def test_capstone_tools_enforce_limit_in_code():
    from ch05.l5_11_1_capstone import issue_large_refund, issue_refund, lookup_order, search_policy
    r = issue_refund.func("ORD-4822", Decimal("25000"), _rt())
    assert r["error_code"] == "NEEDS_APPROVAL"
    assert issue_refund.func("ORD-4821", Decimal("800"), _rt())["success"]
    assert lookup_order.func("ORD-0009", _rt())["error_code"] == "NOT_FOUND"
    assert search_policy.invoke({"query": "approval"}) == ["Refunds above ₹5,000 require human approval."]
    assert len(search_policy.invoke({"query": "zzzz"})) == 3
    assert issue_large_refund.func("ORD-4822", Decimal("25000"), _rt())["success"]


def test_capstone_evals_pass_offline():
    from ch05.l5_11_1_capstone import EVALS, REFUNDS, build_agent, run, tools_called
    from ch05.scripted import ScriptedSupportModel
    agent = build_agent(ScriptedSupportModel())
    for case in EVALS:
        before_refunds = len(REFUNDS)
        result, _ = run(agent, case["msg"], thread_id=f"eval-{case['id']}",
                        user_id=case.get("user", "u_1120"))
        called = tools_called(result["messages"])
        assert case["must_call"] <= called, f"{case['id']} missing: {case['must_call'] - called}"
        assert not (case["must_not_call"] & called), f"{case['id']} forbidden: {case['must_not_call'] & called}"
        if case["id"] in {"bad-order", "not-my-order"}:
            assert len(REFUNDS) == before_refunds, f"{case['id']} mutated REFUNDS"


# ---- Deliberately breaking the 7 capstone guardrails ----
def test_guardrail_1_per_user_ownership():
    from ch05.l5_11_1_capstone import issue_refund, lookup_order
    # Wrong user gets NOT_FOUND for another user's order
    assert lookup_order.func("ORD-4821", _rt("u_9999"))["error_code"] == "NOT_FOUND"
    assert issue_refund.func("ORD-4821", Decimal("800"), _rt("u_9999"))["error_code"] == "NOT_FOUND"


def test_guardrail_2_approval_limit_in_code():
    from ch05.l5_11_1_capstone import issue_refund
    r = issue_refund.func("ORD-4822", Decimal("5001"), _rt())
    assert r["error_code"] == "NEEDS_APPROVAL"


def test_guardrail_3_unknown_order():
    from ch05.l5_11_1_capstone import issue_refund
    r = issue_refund.func("ORD-9999", Decimal("100"), _rt())
    assert r["error_code"] == "NOT_FOUND"


def test_guardrail_4_nonpositive_amount():
    from ch05.l5_11_1_capstone import issue_refund
    for bad in (Decimal("0"), Decimal("-10")):
        r = issue_refund.func("ORD-4821", bad, _rt())
        assert r["error_code"] == "INVALID_AMOUNT"


def test_guardrail_5_exceeds_refundable_balance():
    from ch05.l5_11_1_capstone import issue_refund
    # ORD-4821 total amount is 800
    r = issue_refund.func("ORD-4821", Decimal("1500"), _rt())
    assert r["error_code"] == "EXCEEDS_BALANCE"


def test_guardrail_6_idempotency_prevents_double_refund():
    from ch05.l5_11_1_capstone import REFUNDS, issue_refund
    REFUNDS.clear()
    r1 = issue_refund.func("ORD-4821", Decimal("400"), _rt())
    assert r1["success"] and not r1["data"].get("replayed")
    # Replay with same operation ID returns existing record marked replayed, no duplicate payout
    r2 = issue_refund.func("ORD-4821", Decimal("400"), _rt())
    assert r2["success"] and r2["data"]["replayed"] is True
    # Distinct operation ID allows second genuine partial refund within balance (400 + 400 = 800)
    rt2 = _rt()
    rt2.context.refund_operation_id = "op-2"
    r3 = issue_refund.func("ORD-4821", Decimal("400"), rt2)
    assert r3["success"] and not r3["data"].get("replayed")
    # A third refund exceeds balance (400 + 400 + 100 > 800)
    rt3 = _rt()
    rt3.context.refund_operation_id = "op-3"
    r4 = issue_refund.func("ORD-4821", Decimal("100"), rt3)
    assert r4["error_code"] == "EXCEEDS_BALANCE"


def test_guardrail_7_human_approval_pause_and_rejection():
    from ch05.l5_11_1_capstone import build_agent, run
    m = ScriptedChatModel(script=[ai("", ("issue_large_refund", {"order_id": "ORD-4822", "amount": Decimal("25000")})),
                                  ai("A supervisor declined.")])
    agent = build_agent(m)
    paused, cfg = run(agent, "refund", "t-approval")
    assert "__interrupt__" in paused
    out = agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "no"}]}),
                       config=cfg,
                       context=paused.get("context"))
    assert not any("RF-29382" in str(x.content) for x in out["messages"])


# ---- Part 6 Appendix Budget Guard ----
def test_budget_guard():
    from solutions.ex_6_budget_guard import guard
    escalated = []
    res = guard(steps=16, cost=1.0, case="case1", escalate=escalated.append)
    assert "passed it to a person" in res and escalated == ["case1"]

    res_ok = guard(steps=5, cost=1.0, case="case2", escalate=escalated.append)
    assert res_ok is None and len(escalated) == 1

    res_cost = guard(steps=5, cost=2.5, case="case3", escalate=escalated.append)
    assert "passed it to a person" in res_cost and escalated == ["case1", "case3"]


# ---- exercises: acceptance checks ----
def test_exercise_agent_loop_five_checks():
    from solutions.ex_3_1_5_agent_loop import Call, FakeModel, Response, Tool, run_agent
    assert run_agent(FakeModel([Response(content="hi")]), "x", []) == "hi"
    m = FakeModel([Response(tool_calls=[Call("1", "t", {})]), Response(content="done")])
    assert run_agent(m, "x", [Tool("t", lambda: 1)]) == "done"
    m = FakeModel([Response(tool_calls=[Call("1", "ghost", {})]), Response(content="ok")])
    assert run_agent(m, "x", []) == "ok"

    def boom():
        raise ValueError
    m = FakeModel([Response(tool_calls=[Call("1", "t", {})]), Response(content="ok")])
    assert run_agent(m, "x", [Tool("t", boom)]) == "ok"
    m = FakeModel([Response(tool_calls=[Call("1", "t", {})])])
    assert run_agent(m, "x", [Tool("t", lambda: 1)], max_steps=4).startswith("I wasn't able")
    assert m.i == 4


def test_exercise_starter_is_still_a_starter():
    from exercises.ex_3_1_5_agent_loop_starter import run_agent
    with pytest.raises(NotImplementedError):
        run_agent(None, "x", [])


def test_exercise_score_case_five_checks():
    from solutions.ex_4_eval_scoring import EVALS, score_case
    case = {"must_call": {"a"}, "must_not_call": {"z"}}
    assert score_case(case, {"a"})
    assert not score_case(case, set())
    assert not score_case(case, {"a", "z"})
    assert score_case(case, {"a", "harmless"})
    assert len(EVALS) == 4 and EVALS[3]["must_not_call"]


def test_exercise_refund_window_four_checks():
    from solutions.ex_5_refund_window import issue_refund
    assert issue_refund.invoke({"order_id": "ORD-4821", "amount": 800})["success"]
    assert issue_refund.invoke({"order_id": "ORD-4822", "amount": 25000})["error_code"] == "NEEDS_APPROVAL"
    assert issue_refund.invoke({"order_id": "ORD-4823", "amount": 1200})["error_code"] == "OUTSIDE_WINDOW"
    assert issue_refund.invoke({"order_id": "ORD-0009", "amount": 10})["error_code"] == "NOT_FOUND"


# ---- data files shipped with the repo ----
@pytest.mark.parametrize("p", sorted(DATA.glob("*.json")), ids=lambda p: p.name)
def test_data_files_are_valid_json(p):
    json.loads(p.read_text(encoding="utf-8"))


def test_agent_card_and_memory_record_shapes():
    card = json.loads((DATA / "agent_card.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in card["skills"]] == ["check_stock", "reserve_stock"]
    mem = json.loads((DATA / "memory_record.json").read_text(encoding="utf-8"))
    assert {"id", "user_id", "type", "content", "source", "confidence", "created_at"} <= set(mem)
