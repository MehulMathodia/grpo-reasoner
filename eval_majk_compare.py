"""Compare base vs GRPO adapter under self-consistency (maj@k), the metric GRPO
actually optimizes. Uses a modest test subset + k for a quick, honest read."""

import argparse
import json

from grpo_reasoner.majk import evaluate_majk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default="grpo-adapter-400step")
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=400)
    args = p.parse_args()

    print(f"=== BASE {args.model}  (maj@{args.k}, T={args.temperature}, n={args.limit}) ===")
    base = evaluate_majk(args.model, split="test", limit=args.limit, k=args.k,
                         temperature=args.temperature, batch_size=args.batch_size,
                         max_new_tokens=args.max_new_tokens)
    print(f"  maj@{args.k} = {base.majk_accuracy:.1%} | mean pass-rate = {base.mean_pass_rate:.1%}")

    print(f"=== GRPO adapter {args.adapter} ===")
    grpo = evaluate_majk(args.model, adapter_path=args.adapter, split="test", limit=args.limit,
                         k=args.k, temperature=args.temperature, batch_size=args.batch_size,
                         max_new_tokens=args.max_new_tokens)
    print(f"  maj@{args.k} = {grpo.majk_accuracy:.1%} | mean pass-rate = {grpo.mean_pass_rate:.1%}")

    print("\n=== DELTA ===")
    print(json.dumps({
        "base_majk": base.majk_accuracy, "grpo_majk": grpo.majk_accuracy,
        "delta_majk": grpo.majk_accuracy - base.majk_accuracy,
        "base_pass_rate": base.mean_pass_rate, "grpo_pass_rate": grpo.mean_pass_rate,
        "delta_pass_rate": grpo.mean_pass_rate - base.mean_pass_rate,
    }, indent=2))


if __name__ == "__main__":
    main()
