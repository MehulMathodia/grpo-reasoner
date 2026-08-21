"""GSM8K loading and prompt formatting.

GSM8K is a benchmark of grade-school math word problems. Each example has a
free-text `question` and an `answer` that ends with a line `#### <number>`
giving the final numeric answer. That `#### <number>` convention is what makes
the reward *verifiable*: we can extract the gold number exactly and check the
model's final answer against it, with no judge model or heuristic in the loop.
"""

import re

from datasets import load_dataset

# The system prompt defines the output contract the reward functions score
# against: think step by step, then give the final answer inside \boxed{}.
# Keeping the format explicit and simple is what lets the format reward be a
# clean, unambiguous signal rather than a fuzzy one.
SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, showing your "
    "reasoning. Then give the final numeric answer on its own line in the form "
    "\\boxed{ANSWER}."
)

_GOLD_RE = re.compile(r"####\s*(-?[0-9][0-9,]*)")


def extract_gold_answer(answer_field: str) -> str:
    """Pull the gold final number out of a GSM8K `answer` field."""
    m = _GOLD_RE.search(answer_field)
    if not m:
        raise ValueError(f"No '#### <number>' found in answer: {answer_field!r}")
    return m.group(1).replace(",", "")


def build_prompt(question: str) -> list[dict]:
    """Chat-format prompt for a single question (what the policy model sees)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def load_gsm8k(split: str = "train", limit: int | None = None):
    """Load GSM8K and attach `prompt` (chat messages) + `gold` (string number).

    Returns a HuggingFace Dataset with columns: question, gold, prompt.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    def _map(ex):
        return {
            "gold": extract_gold_answer(ex["answer"]),
            "prompt": build_prompt(ex["question"]),
        }

    return ds.map(_map, remove_columns=[c for c in ds.column_names if c != "question"])
