"""Listing 3.6.3 - Self-correction: retry with the validation error in the prompt.

Run:  python -m ch03.l3_6_3_generate_valid
"""
from pydantic import BaseModel, ValidationError


class OutputSchema(BaseModel):  # the book assumes this exists
    answer: str
    confidence: float


def generate_valid(model, messages, max_attempts: int = 3):
    for _ in range(max_attempts):
        result = model.invoke(messages)
        try:
            return OutputSchema.model_validate_json(result.content)
        except ValidationError as err:
            messages += [
                result,
                {"role": "user",
                 "content": f"That failed validation: {err}. Return corrected JSON only."},
            ]
    raise RuntimeError("No valid output after retries; escalate to a human.")


if __name__ == "__main__":
    from common.fakes import TextModel
    m = TextModel(['{"answer": "x"}', '{"answer": "Paris", "confidence": 0.9}'])
    print(generate_valid(m, [{"role": "user", "content": "Capital of France? JSON only."}]))
