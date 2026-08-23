"""Rejection-sampling fine-tuning (RFT) data: the strongest *fair* SFT baseline
for a GRPO comparison.

Plain SFT on GSM8K's human-written rationales is off-policy: those rationales
are terse and stylistically far from how an instruct model already reasons, so
imitating them can make a strong model worse. RFT removes that confound: sample
solutions from the base model itself, keep only the ones whose final answer is
correct (verified with the same exact-match reward GRPO uses), and SFT on those.
It is on-policy supervised learning with the same verifiable signal — so the
comparison GRPO-vs-RFT isolates "RL update" from "on-policy data".
"""

import json
import random

from grpo_reasoner.data import load_gsm8k
from grpo_reasoner.generate import load_model, generate_batch
from grpo_reasoner.rewards import extract_answer, _normalize_number


def build_rft_dataset(
    model_id: str,
    n_prompts: int = 600,
    k: int = 4,
    temperature: float = 0.7,
    max_per_prompt: int = 2,
    max_new_tokens: int = 512,
    batch_size: int = 32,
    seed: int = 0,
    out_path: str | None = None,
) -> list[dict]:
    """Sample `k` solutions per prompt from `model_id` on `n_prompts` GSM8K train
    questions (seeded shuffle), keep up to `max_per_prompt` *correct* ones, and
    return prompt/completion records (write them to `out_path` as JSONL if given)."""
    ds = load_gsm8k(split="train").shuffle(seed=seed).select(range(n_prompts))
    prompts, golds = list(ds["prompt"]), list(ds["gold"])

    model, tokenizer = load_model(model_id)
    expanded = [p for p in prompts for _ in range(k)]
    completions = generate_batch(model, tokenizer, expanded, max_new_tokens=max_new_tokens,
                                 temperature=temperature, batch_size=batch_size)

    rng = random.Random(seed)
    records, n_correct_total = [], 0
    for i, (prompt, gold) in enumerate(zip(prompts, golds)):
        group = completions[i * k:(i + 1) * k]
        correct = [c for c in group if extract_answer(c) == _normalize_number(gold)]
        n_correct_total += len(correct)
        # dedupe identical completions, cap per prompt to avoid over-weighting easy items
        uniq = list(dict.fromkeys(correct))
        rng.shuffle(uniq)
        for c in uniq[:max_per_prompt]:
            records.append({"prompt": prompt, "completion": [{"role": "assistant", "content": c}]})

    stats = {"n_prompts": n_prompts, "k": k, "sampled": n_prompts * k,
             "correct_samples": n_correct_total, "kept": len(records),
             "prompts_with_any_correct": sum(1 for i in range(n_prompts)
                                              if any(extract_answer(c) == _normalize_number(golds[i])
                                                     for c in completions[i*k:(i+1)*k]))}
    print("RFT data:", json.dumps(stats), flush=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        with open(out_path.replace(".jsonl", "_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    return records
