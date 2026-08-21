"""Rigorous evaluation harness: exact-match accuracy on the GSM8K test set.

The whole point of this project is an *honest* before/after number, so this
harness is deliberately strict and reproducible:

  * greedy decoding (temperature 0) — deterministic, no lucky-sample inflation;
  * the SAME answer extractor used in training rewards, so eval and reward agree;
  * evaluation on the held-out `test` split, never on anything seen in training;
  * per-example results saved so improvements can be analyzed, not just asserted.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from grpo_reasoner.data import load_gsm8k
from grpo_reasoner.generate import load_model, generate_batch
from grpo_reasoner.rewards import extract_answer, _normalize_number


@dataclass
class EvalResult:
    model_id: str
    adapter_path: str | None
    n: int
    accuracy: float
    n_parseable: int          # how many completions produced any extractable answer
    results_path: str | None


def evaluate_items(
    model_id: str,
    items: list[dict],
    adapter_path: str | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    save_path: str | None = None,
    matcher=None,
) -> EvalResult:
    """Generic greedy evaluation over `items` = [{'question': str, 'gold': str, ...}].
    `matcher(pred, gold) -> bool` defaults to GSM8K's exact normalized string
    match; transfer benchmarks pass a numeric-tolerant matcher instead. Same
    extractor, same decoding, same output format as `evaluate`."""
    from grpo_reasoner.data import build_prompt

    if matcher is None:
        matcher = lambda pred, gold: pred is not None and pred == _normalize_number(gold)  # noqa: E731

    model, tokenizer = load_model(model_id, adapter_path=adapter_path)
    prompts = [build_prompt(it["question"]) for it in items]
    completions = generate_batch(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens, temperature=0.0, batch_size=batch_size,
    )

    records, n_correct, n_parseable = [], 0, 0
    for it, comp in zip(items, completions):
        pred = extract_answer(comp)
        correct = bool(matcher(pred, it["gold"]))
        n_correct += int(correct)
        n_parseable += int(pred is not None)
        rec = {**{k: v for k, v in it.items() if k != "prompt"},
               "pred": pred, "correct": correct, "completion": comp}
        records.append(rec)

    results_path = None
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        results_path = save_path

    return EvalResult(
        model_id=model_id, adapter_path=adapter_path, n=len(records),
        accuracy=n_correct / len(records) if records else 0.0,
        n_parseable=n_parseable, results_path=results_path,
    )


def evaluate(
    model_id: str,
    adapter_path: str | None = None,
    split: str = "test",
    limit: int | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    save_path: str | None = None,
) -> EvalResult:
    ds = load_gsm8k(split=split, limit=limit)
    model, tokenizer = load_model(model_id, adapter_path=adapter_path)

    prompts = list(ds["prompt"])
    golds = list(ds["gold"])
    questions = list(ds["question"])

    completions = generate_batch(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens, temperature=0.0, batch_size=batch_size,
    )

    records, n_correct, n_parseable = [], 0, 0
    for q, gold, comp in zip(questions, golds, completions):
        pred = extract_answer(comp)
        correct = pred is not None and pred == _normalize_number(gold)
        n_correct += int(correct)
        n_parseable += int(pred is not None)
        records.append({"question": q, "gold": gold, "pred": pred, "correct": correct, "completion": comp})

    results_path = None
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        results_path = save_path

    return EvalResult(
        model_id=model_id,
        adapter_path=adapter_path,
        n=len(records),
        accuracy=n_correct / len(records) if records else 0.0,
        n_parseable=n_parseable,
        results_path=results_path,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate a model on GSM8K exact-match accuracy.")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="Optional path to a trained LoRA adapter.")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None, help="Evaluate on the first N examples only.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--save", default=None, help="Path to save per-example JSONL results.")
    args = p.parse_args()

    res = evaluate(
        model_id=args.model, adapter_path=args.adapter, split=args.split,
        limit=args.limit, max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size, save_path=args.save,
    )
    print(json.dumps(asdict(res), indent=2))
    print(f"\nAccuracy: {res.accuracy:.1%}  ({res.n} examples, {res.n_parseable} parseable)")
