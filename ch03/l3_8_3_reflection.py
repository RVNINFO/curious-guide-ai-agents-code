"""Listing 3.8.3 - Reflection pattern: generate, critique, revise (bounded rounds).

Run:  python -m ch03.l3_8_3_reflection
"""
CRITERIA = "polite, under 80 words, mentions the refund reference number"


def make_with_reflection(generator, critic):
    def with_reflection(task, max_rounds=2):
        draft = generator.invoke(task)
        for _ in range(max_rounds):
            critique = critic.invoke(
                f"Task: {task}\n\nDraft:\n{draft}\n\n"
                f"Evaluate against these criteria: {CRITERIA}. "
                f"If it fully meets them, reply exactly: APPROVED. "
                f"Otherwise list specific, actionable problems."
            )
            if critique.strip().startswith("APPROVED"):
                break
            draft = generator.invoke(
                f"Task: {task}\n\nYour draft:\n{draft}\n\n"
                f"Feedback:\n{critique}\n\nProduce an improved version."
            )
        return draft

    return with_reflection


class _Str:
    """Adapts a list of replies to the `.invoke(prompt) -> str` interface the listing expects."""

    def __init__(self, replies):
        self.replies, self.i, self.prompts = replies, 0, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return r


if __name__ == "__main__":
    gen = _Str(["Refund done.", "Hi! Your refund RF-29381 is processed."])
    critic = _Str(["Missing the reference number.", "APPROVED"])
    print(make_with_reflection(gen, critic)("Write the refund confirmation."))
