"""Transfer evaluation on benchmarks the adapter never trained on.

The GSM8K gain could be (a) better general math reasoning or (b) overfitting to
GSM8K's style. Two held-out benchmarks separate those:

  * SVAMP  — 1,000 elementary word problems from a different source/distribution,
             built specifically to defeat shallow pattern-matching.
  * GSM-Hard — the GSM8K questions with the numbers replaced by large/awkward
             values. Same reasoning, harder arithmetic: isolates arithmetic
             robustness from problem understanding.

Answers on both are numeric but not always clean integers (GSM-Hard targets are
floats, often negative or 7+ digits), so grading uses a numeric-tolerant match
rather than the string match that is exact for GSM8K.
"""

import math

from datasets import load_dataset

from grpo_reasoner.rewards import extract_answer


def _to_float(s) -> float | None:
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def answers_match(pred: str | None, gold, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> bool:
    """Numeric comparison: `42` == `42.0` == `42.00`, `1,234` == 1234.0,
    `-9,867,630` == -9867630.0. Tolerances absorb float formatting only —
    they are far tighter than any two distinct plausible answers."""
    if pred is None:
        return False
    p, g = _to_float(pred), _to_float(gold)
    if p is None or g is None:
        return False
    return math.isclose(p, g, rel_tol=rel_tol, abs_tol=abs_tol)


def load_svamp(limit: int | None = None) -> list[dict]:
    """All 1,000 SVAMP problems (train+test — none were used in training)."""
    ds = load_dataset("ChilleD/SVAMP")
    items = []
    for split in ("train", "test"):
        for r in ds[split]:
            q = f"{r['Body'].strip()} {r['Question'].strip()}"
            items.append({"question": q, "gold": str(r["Answer"]), "source": f"svamp-{split}"})
    return items[:limit] if limit else items


def load_gsm_hard(limit: int | None = None) -> list[dict]:
    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    items = [{"question": r["input"], "gold": str(r["target"]), "source": "gsm-hard"} for r in ds]
    return items[:limit] if limit else items


def grade(completion: str, gold) -> tuple[str | None, bool]:
    pred = extract_answer(completion)
    return pred, answers_match(pred, gold)
