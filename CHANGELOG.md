# Changelog

A plain record of what was done, in order, so the history of this project stays
honest. Each entry states what was actually run and what was actually measured —
no rounding up.

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
