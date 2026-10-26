"""Listing 3.8.5 - Reviewer pattern: deterministic checks first, then a model, then policy.

Run:  python -m ch03.l3_8_5_reviewer
"""
from dataclasses import dataclass


@dataclass
class Action:
    tool: str
    args: dict


@dataclass
class Verdict:
    ok: bool
    reason: str = ""


@dataclass
class Rejected:
    reason: str


@dataclass
class Escalate:
    reason: str


@dataclass
class Approved:
    pass


@dataclass
class Judgment:
    decision: str  # "approve" | "reject" | "uncertain"
    reason: str = ""


def amount_is_positive(action, context) -> Verdict:
    amount = action.args.get("amount", 1)
    return Verdict(amount > 0, "amount must be positive")


DETERMINISTIC_CHECKS = [amount_is_positive]
HIGH_STAKES = {"issue_refund"}
ALWAYS_HUMAN = {"delete_account"}
REVIEW_PROMPT = "Review this action.\nAction: {action}\nContext: {context}"


def make_review(reviewer_model):
    def review(action, context):
        for check in DETERMINISTIC_CHECKS:  # cheap and certain
            verdict = check(action, context)
            if not verdict.ok:
                return Rejected(verdict.reason)

        if action.tool in HIGH_STAKES:
            judgment = reviewer_model.invoke(
                REVIEW_PROMPT.format(action=action, context=context))
            if judgment.decision == "reject":
                return Rejected(judgment.reason)
            if judgment.decision == "uncertain":
                return Escalate(judgment.reason)

        if action.tool in ALWAYS_HUMAN:
            return Escalate("policy: requires human approval")

        return Approved()

    return review


class _FixedReviewer:
    def __init__(self, decision):
        self.decision, self.calls = decision, 0

    def invoke(self, prompt):
        self.calls += 1
        return Judgment(self.decision, "because")


if __name__ == "__main__":
    review = make_review(_FixedReviewer("approve"))
    print(review(Action("issue_refund", {"amount": 50}), {}))
    print(review(Action("issue_refund", {"amount": -1}), {}))
    print(review(Action("delete_account", {}), {}))
