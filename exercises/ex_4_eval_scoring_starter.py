"""Exercise starter (end of Part 4): score eval cases on tool use."""


def score_case(case: dict, called: set) -> bool:
    """TODO: True if every tool in case["must_call"] was called
    and no tool in case["must_not_call"] was called."""
    raise NotImplementedError


EVALS = [
    {
        "id": "small-damaged",
        "msg": "ORD-4821 arrived damaged. Refund please.",
        "must_call": {"lookup_order", "search_policy", "issue_refund"},
        "must_not_call": {"issue_large_refund"}
    },
    {
        "id": "large-damaged",
        "msg": "ORD-4822 arrived damaged. Refund please.",
        "must_call": {"lookup_order", "issue_large_refund"},
        "must_not_call": set()
    },
    {
        "id": "bad-order",
        "msg": "Refund ORD-0009, it broke.",
        "must_call": {"lookup_order"},
        "must_not_call": {"issue_refund", "issue_large_refund"}
    },
    # TODO (part b): add a fourth case for a refund request on an order
    # delivered 45 days ago. Which tool must NOT be called?
]
