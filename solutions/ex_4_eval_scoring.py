"""Reference solution for the Part 4 exercise (eval scoring)."""
from exercises.ex_4_eval_scoring_starter import EVALS as _STARTER_EVALS


def score_case(case: dict, called: set) -> bool:
    return case["must_call"] <= called and not (case["must_not_call"] & called)


EVALS = _STARTER_EVALS + [
    {
        "id": "expired-window",
        "msg": "ORD-4823 arrived damaged. Refund please.",  # delivered 45 days ago
        "must_call": {"lookup_order"},
        "must_not_call": {"issue_refund", "issue_large_refund"},  # outside the 30-day window
    },
]
