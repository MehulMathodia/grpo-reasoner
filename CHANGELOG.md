# Changelog

A plain record of what was done, in order, so the history of this project stays
honest. Each entry states what was actually run and what was actually measured —
no rounding up.

## v0.2.0 — 2026-08-23 — transfer evaluation on unseen benchmarks

**What was added**
- `grpo_reasoner/transfer.py`: SVAMP (all 1,000) and GSM-Hard loaders, numeric-
  tolerant grading (floats, negatives, thousands separators) — 6 new tests, 23 total.
- `eval.evaluate_items`: generic paired greedy eval with a pluggable matcher.
- `notebooks/kaggle_transfer_kernel.py`: the kernel that produced the numbers
  (Kaggle P100, 5.4 h).

**What was run and what it showed** — same v0.1 adapter, no retraining:
- GSM-Hard (n=1,300): 46.0% → 50.2% (+4.2), 144 fixed / 89 regressed, p = 0.0004.
- SVAMP (n=1,000): 81.2% → 83.7% (+2.5), 90 fixed / 65 regressed, **p = 0.054 —
  borderline, not claimed as significant.**
- Pooled with GSM8K over 3,619 paired questions: 363 fixed / 251 regressed, p ≈ 7e-6.
- GSM-Hard used 1,300 of 1,319 questions (the kernel sized it to its time budget).

**What this does and doesn't show**
- The gain is not GSM8K-style overfitting: it moves two held-out benchmarks in the
  same direction, most strongly the arithmetic-heavy one.
- It does not show general reasoning improvement beyond grade-school-style math;
  nothing harder (e.g. MATH) was evaluated. Single seed still.

## v0.1.0 — 2026-08-19 — first public release

**What exists**
- Reward functions (verifiable exact-match + small format shaping), eval harness
  (greedy, held-out test only, per-question JSONL), self-consistency (maj@k) eval,
  GRPO training script (TRL `GRPOTrainer` + LoRA, `beta=0`, `dr_grpo`), Gradio demo.
- 17 unit tests for reward/data logic (no GPU needed).

**What was run and what it showed**
- *Qwen2.5-0.5B-Instruct, 400 steps, local RTX 5060 (8GB):* training reward rose
  0.2 → 0.55, but held-out GSM8K did **not** improve — greedy 43.6% → 44.8% on n=500
  (noise), maj@8 58.7% → 54.7% on n=150 (noise, negative). Treated as a **negative
  result**; adapter not shipped (its checkpoint dir is ~0.5 GB).
- *Qwen2.5-1.5B-Instruct, 300 steps, Kaggle P100 (free tier), no vLLM:* greedy
  accuracy on the **full 1,319-question test split** 69.14% → 71.57%
  (+2.43 pts; 129 fixed / 97 regressed; exact McNemar p = 0.039). maj@8 on n=100:
  79% → 78% (flat). **This is the shipped adapter** (`adapter-1.5b/`).

**Known limitations (stated up front)**
- The setup (GRPO on GSM8K with Qwen2.5) is the standard tutorial configuration;
  the contribution here is the evaluation rigor and the documented negative result,
  not a novel method.
- +2.4 points is a modest gain; p = 0.039 is significant but not overwhelming.
- The gain is in greedy decoding only; maj@8 did not move.
- Single training seed; no ablations yet (format reward, KL, SFT baseline).
- Total compute: ~1.5 h local (0.5B) + ~5.3 h Kaggle P100 training + ~5 h Kaggle
  full-test eval.

**Not yet done** (candidates for the next entries): transfer eval on a second
benchmark, GRPO-vs-SFT ablation, a second seed, public HF Space.
