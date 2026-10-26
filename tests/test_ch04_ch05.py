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
    assert "CUSTOMER ASKED:\nq" in m.calls[0] and '"grounded": n' in m.calls[0]  # braces survive .format
    assert "{question}" in JUDGE_PROMPT


def test_process_scoring_checks_tools_and_answer():
    from ch04.l4_2_3_llm_judge import load_case, score_process
    case = load_case()
    good = {"get_order_status", "start_return"}
    assert score_process(case, good, "You can return it within 30 days.")
    assert not score_process(case, good | {"issue_refund"}, "return within 30 days")   # forbidden tool
    assert not score_process(case, {"get_order_status"}, "return within 30 days")       # missing tool
    assert not score_process(case, good, "refund approved, return in 30 days")          # forbidden phrase


# ---- Part 5: tenant-scoped refund ----
def test_cross_tenant_refund_is_denied():
    """The book's pytest-style listing, run against a real harness."""
    from ch05.harness import run_agent
    result = run_agent(
        request="Check order ORD-4202 for a refund",
        test_identity={"tenant_id": "tenant-a"},
        fixtures={"ORD-4202": {"tenant_id": "tenant-b"}},
    )
    assert result.status == "denied"
    assert result.cross_tenant_records_returned == 0
    from ch05 import adapters
    assert adapters.payments.calls == []  # no money moved
    # the model DID try; the tool, not the model, is what stopped it
    assert result.tool_calls_named("issue_refund") == 1


def test_same_tenant_refund_succeeds_and_is_idempotent():
    from ch05 import adapters
    from ch05.harness import run_agent
    r = run_agent("Refund ORD-1", {"tenant_id": "tenant-a"}, {"ORD-1": {"tenant_id": "tenant-a"}})
    assert r.status == "completed" and len(adapters.payments.calls) == 1
    key = adapters.payments.calls[0]["idempotency_key"]
    assert key == "refund:tenant-a:test-op-1"
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


# ---- Part 5: capstone 5.11.1 ----
def test_capstone_tools_enforce_limit_in_code():
    from ch05.l5_11_1_capstone import issue_large_refund, issue_refund, lookup_order, search_policy
    r = issue_refund.invoke({"order_id": "ORD-4822", "amount": 25000})
    assert r["error_code"] == "NEEDS_APPROVAL"       # the limit is not left to the prompt
    assert issue_refund.invoke({"order_id": "ORD-4821", "amount": 800})["success"]
    assert lookup_order.invoke({"order_id": "ORD-0009"})["error_code"] == "NOT_FOUND"
    assert search_policy.invoke({"query": "approval"}) == ["Refunds above ₹5,000 require human approval."]
    assert len(search_policy.invoke({"query": "zzzz"})) == 3  # falls back to the full policy
    assert issue_large_refund.invoke({"order_id": "ORD-4822", "amount": 25000})["success"]


def test_capstone_evals_pass_offline():
    from ch05.l5_11_1_capstone import run_evals
    from ch05.scripted import scripted_agent_for_case
    assert run_evals(None, agent_for_case=scripted_agent_for_case) == {
        "small-damaged": True, "large-damaged": True, "bad-order": True}


def test_capstone_eval_scoring_detects_failures():
    """An eval harness that cannot fail is worthless: feed it a misbehaving model."""
    from ch05.l5_11_1_capstone import build_agent, run_evals
    bad = lambda case: build_agent(ScriptedChatModel(script=[  # refunds without ever checking the order
        ai("", ("issue_refund", {"order_id": "ORD-4821", "amount": 800})), ai("done")]))
    res = run_evals(None, agent_for_case=bad)
    assert res["small-damaged"] is False and res["bad-order"] is False


def test_capstone_large_refund_pauses_until_human_approves_then_runs():
    from ch05.l5_11_1_capstone import approve, build_agent, run, tools_called
    m = ScriptedChatModel(script=[ai("", ("lookup_order", {"order_id": "ORD-4822"}),
                                         ("search_policy", {"query": "approval"})),
                                  ai("", ("issue_large_refund", {"order_id": "ORD-4822", "amount": 25000})),
                                  ai("Approved and refunded, ref RF-29382.")])
    agent = build_agent(m)
    paused, cfg = run(agent, "ORD-4822 arrived damaged. Refund please.", "t1")
    assert "__interrupt__" in paused
    assert not any(getattr(x, "type", "") == "tool" and "RF-29382" in str(x.content)
                   for x in paused["messages"])  # the refund has NOT executed yet
    done = approve(agent, cfg)
    assert any("RF-29382" in str(x.content) for x in done["messages"] if getattr(x, "type", "") == "tool")


def test_capstone_rejection_blocks_refund():
    from ch05.l5_11_1_capstone import build_agent, run
    m = ScriptedChatModel(script=[ai("", ("issue_large_refund", {"order_id": "ORD-4822", "amount": 25000})),
                                  ai("A supervisor declined.")])
    agent = build_agent(m)
    _, cfg = run(agent, "refund", "t2")
    out = agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "no"}]}), config=cfg)
    assert not any("RF-29382" in str(x.content) for x in out["messages"])


# ---- exercises: the book's stated acceptance checks ----
def test_exercise_agent_loop_five_checks():
    from solutions.ex_3_1_5_agent_loop import Call, FakeModel, Response, Tool, run_agent
    assert run_agent(FakeModel([Response(content="hi")]), "x", []) == "hi"                          # 1
    m = FakeModel([Response(tool_calls=[Call("1", "t", {})]), Response(content="done")])
    assert run_agent(m, "x", [Tool("t", lambda: 1)]) == "done"                                      # 2
    m = FakeModel([Response(tool_calls=[Call("1", "ghost", {})]), Response(content="ok")])
    assert run_agent(m, "x", []) == "ok"                                                            # 3

    def boom():
        raise ValueError
    m = FakeModel([Response(tool_calls=[Call("1", "t", {})]), Response(content="ok")])
    assert run_agent(m, "x", [Tool("t", boom)]) == "ok"                                             # 4
    m = FakeModel([Response(tool_calls=[Call("1", "t", {})])])
    assert run_agent(m, "x", [Tool("t", lambda: 1)], max_steps=4).startswith("I wasn't able")       # 5
    assert m.i == 4


def test_exercise_starter_is_still_a_starter():
    from exercises.ex_3_1_5_agent_loop_starter import run_agent
    with pytest.raises(NotImplementedError):
        run_agent(None, "x", [])


def test_exercise_score_case_five_checks():
    from solutions.ex_4_eval_scoring import EVALS, score_case
    case = {"must_call": {"a"}, "must_not_call": {"z"}}
    assert score_case(case, {"a"})                      # right tools
    assert not score_case(case, set())                  # missing required
    assert not score_case(case, {"a", "z"})             # forbidden called
    assert score_case(case, {"a", "harmless"})          # extras ignored
    assert len(EVALS) == 4 and EVALS[3]["must_not_call"]  # 4th case has a forbidden tool


def test_exercise_refund_window_four_checks():
    from solutions.ex_5_refund_window import issue_refund
    assert issue_refund.invoke({"order_id": "ORD-4821", "amount": 800})["success"]
    assert issue_refund.invoke({"order_id": "ORD-4822", "amount": 25000})["error_code"] == "NEEDS_APPROVAL"
    assert issue_refund.invoke({"order_id": "ORD-4823", "amount": 1200})["error_code"] == "OUTSIDE_WINDOW"
    assert issue_refund.invoke({"order_id": "ORD-0009", "amount": 10})["error_code"] == "NOT_FOUND"


# ---- data files shipped with the repo must stay valid JSON with the shapes the book describes ----
@pytest.mark.parametrize("p", sorted(DATA.glob("*.json")), ids=lambda p: p.name)
def test_data_files_are_valid_json(p):
    json.loads(p.read_text(encoding="utf-8"))


def test_agent_card_and_memory_record_shapes():
    card = json.loads((DATA / "agent_card.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in card["skills"]] == ["check_stock", "reserve_stock"]
    mem = json.loads((DATA / "memory_record.json").read_text(encoding="utf-8"))
    assert {"id", "user_id", "type", "content", "source", "confidence", "created_at"} <= set(mem)
