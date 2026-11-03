"""Listing 4.2.9 - When the same input gives different answers: run each eval case n times, report pass rates.
Run:  python -m ch04.l4_2_9_pass_rates
"""


def score_case(outcomes: list[bool]) -> dict:
    """outcomes holds one True/False per run of the same case."""
    passes = sum(outcomes)
    return {
        "pass_rate": passes / len(outcomes),
        "pass_at_n": passes > 0,  # did at least one run succeed?
        "all_n": passes == len(outcomes),  # did every run succeed?
    }


# Five runs per case
# True means the case's checks passed on that run.
runs = {
    "cross_tenant_refund_denied": [True, True, True, True, True],
    "refund_within_policy": [True, True, False, True, True],
    "ambiguous_order_id": [False, True, False, False, True],
}

if __name__ == "__main__":
    for case_id, outcomes in runs.items():
        print(case_id, score_case(outcomes))

    overall = (sum(sum(o) for o in runs.values())
               / sum(len(o) for o in runs.values()))
    print(f"overall pass rate = {overall:.2f}")
    # overall pass rate = 0.73
