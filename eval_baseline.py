"""Measure honest baselines for candidate base models on GSM8K, before any
training. Run once to decide which base model gives the best headroom for GRPO
to demonstrate a gain.

Usage: python eval_baseline.py --limit 150
"""

import argparse
import json

from grpo_reasoner.eval import evaluate

CANDIDATES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=150, help="Number of GSM8K test examples.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    rows = []
    for model_id in CANDIDATES:
        print(f"\n=== {model_id} on {args.limit} GSM8K test examples ===")
        res = evaluate(
            model_id=model_id, split="test", limit=args.limit,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            save_path=f"results/baseline_{model_id.split('/')[-1]}.jsonl",
        )
        print(f"  accuracy = {res.accuracy:.1%}  ({res.n_parseable}/{res.n} parseable)")
        rows.append({"model": model_id, "accuracy": res.accuracy, "n": res.n, "parseable": res.n_parseable})

    print("\n=== SUMMARY ===")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
