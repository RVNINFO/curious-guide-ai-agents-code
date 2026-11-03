"""Listing 4.2.3 - LLM-as-judge prompt + a scorer, and the 4.2.2 eval-case scorer.

Offline: pass any model whose .invoke(prompt) returns an object with .content (JSON string).
Live:    ANTHROPIC_API_KEY=... python -m ch04.l4_2_3_llm_judge
"""
import json
from pathlib import Path

JUDGE_PROMPT = """
You are evaluating a customer support response.

CUSTOMER ASKED:
{question}

SOURCE MATERIAL AVAILABLE TO THE AGENT:
{context}

AGENT RESPONDED:
{answer}

Score each criterion 1-5 and give a one-sentence justification.

1. GROUNDED  - every factual claim is supported by the source material
2. COMPLETE  - fully answers what was asked
3. CORRECT   - no factual errors
4. TONE      - warm, clear, free of corporate filler

Return JSON: {{"grounded": n, "complete": n, "correct": n,
               "tone": n, "notes": "..."}}
"""


def judge(model, question: str, context: str, answer: str) -> dict:
    reply = model.invoke(JUDGE_PROMPT.format(question=question, context=context, answer=answer))
    return json.loads(reply.content)


def load_case(name: str = "eval_case_refund_within_policy.json") -> dict:
    return json.loads((Path(__file__).parent.parent / "data" / name).read_text(encoding="utf-8"))


def score_process(case: dict, tools_called: set, answer: str) -> bool:
    """Process-and-answer check from 4.2.2: right tools, forbidden tools, required phrases."""
    e = case["expects"]
    return (
        set(e["should_call_tools"]) <= tools_called
        and not (set(e["should_not_call_tools"]) & tools_called)
        and all(p.lower() in answer.lower() for p in e["answer_must_mention"])
        and not any(p.lower() in answer.lower() for p in e["answer_must_not_mention"])
    )


if __name__ == "__main__":
    from langchain.chat_models import init_chat_model
    m = init_chat_model("anthropic:claude-sonnet-5-5")
    print(judge(m, "Can I return my kettle?", "Policy: 30-day returns.", "Yes, within 30 days."))
