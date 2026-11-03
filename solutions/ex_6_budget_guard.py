"""Reference for the Part 6 appendix answer: stop a run that exceeds its step or cost budget and hand it to a person.
The book prints only the guard (`if steps >= MAX_STEPS or cost >= MAX_COST: ...`); this wraps it so it can run.
"""
MAX_STEPS = 15
MAX_COST = 2.00  # dollars per task


def guard(steps: int, cost: float, case, escalate):
    """Return a hand-off message if the budget is spent, else None."""
    if steps >= MAX_STEPS or cost >= MAX_COST:
        escalate(case)
        return ("I couldn't complete this, "
                "so I've passed it to a person.")
    return None
