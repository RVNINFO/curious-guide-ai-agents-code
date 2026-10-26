"""Scripted 'model behaviour' for the capstone evals, so they run offline and deterministically.
These scripts encode what a correct model should do; they test the *harness and guardrails*
(tools, approval pause, eval scoring), not any real model's judgment. Live evals use --live.
"""
from ch05.l5_11_1_capstone import build_agent
from common.fakes import ScriptedChatModel, ai

SCRIPTS = {
    "small-damaged": [
        ai("", ("lookup_order", {"order_id": "ORD-4821"}),
               ("search_policy", {"query": "damaged refund"})),
        ai("", ("issue_refund", {"order_id": "ORD-4821", "amount": 800})),
        ai("Your refund of ₹800 has been processed. Reference RF-29381."),
    ],
    "large-damaged": [
        ai("", ("lookup_order", {"order_id": "ORD-4822"}),
               ("search_policy", {"query": "refund approval"})),
        ai("", ("issue_large_refund", {"order_id": "ORD-4822", "amount": 25000})),
        ai("Your refund of ₹25000 was approved and processed."),
    ],
    "bad-order": [
        ai("", ("lookup_order", {"order_id": "ORD-0009"})),
        ai("I couldn't find that order. Could you check the ID?"),
    ],
}


def scripted_agent_for_case(case):
    return build_agent(ScriptedChatModel(script=SCRIPTS[case["id"]]))
