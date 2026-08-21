"""Self-consistency (maj@k) evaluation.

GRPO optimizes the *sampling* distribution — it makes correct completions more
probable within a sampled group. The metric that most directly reflects that is
**self-consistency**: sample k completions at a nonzero temperature, extract each
final answer, and take the majority vote (Wang et al., 2022). This is a standard,
honest GSM8K metric — arguably the *right* one for a model trained with GRPO,
where greedy decoding can undersell the gain because it ignores the sharpened
distribution around the correct answer.
"""

from collections import Counter
from dataclasses import dataclass

from grpo_reasoner.data import load_gsm8k
from grpo_reasoner.generate import load_model, generate_batch
from grpo_reasoner.rewards import extract_answer, _normalize_number


@dataclass
class MajKResult:
    model_id: str
    adapter_path: str | None
    n: int
    k: int
    temperature: float
    majk_accuracy: float       # majority-vote accuracy
    mean_pass_rate: float      # mean fraction of the k samples that were correct (pass@1 under sampling)


def _majority_answer(answers: list[str | None]) -> str | None:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def evaluate_majk(
    model_id: str,
    adapter_path: str | None = None,
    split: str = "test",
    limit: int | None = None,
    k: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    batch_size: int = 16,
) -> MajKResult:
    ds = load_gsm8k(split=split, limit=limit)
    model, tokenizer = load_model(model_id, adapter_path=adapter_path)

    golds = list(ds["gold"])
    prompts = list(ds["prompt"])

    # replicate each prompt k times, generate all at temperature T, then regroup
    expanded = [p for p in prompts for _ in range(k)]
    completions = generate_batch(
        model, tokenizer, expanded,
        max_new_tokens=max_new_tokens, temperature=temperature, batch_size=batch_size,
    )

    n_majk_correct = 0
    total_pass = 0.0
    for i, gold in enumerate(golds):
        group = completions[i * k : (i + 1) * k]
        preds = [extract_answer(c) for c in group]
        gnorm = _normalize_number(gold)
        # majority vote
        maj = _majority_answer(preds)
        n_majk_correct += int(maj is not None and maj == gnorm)
        # per-sample pass rate (pass@1 under sampling)
        total_pass += sum(1 for p in preds if p is not None and p == gnorm) / k

    n = len(golds)
    return MajKResult(
        model_id=model_id,
        adapter_path=adapter_path,
        n=n,
        k=k,
        temperature=temperature,
        majk_accuracy=n_majk_correct / n if n else 0.0,
        mean_pass_rate=total_pass / n if n else 0.0,
    )
