"""Post-training results: evaluate base vs GRPO-trained model on the GSM8K test
split, and render (1) the before/after accuracy comparison and (2) the training
reward curve parsed from the training log.

Run after train_grpo.py finishes:
    python plot_results.py --adapter grpo-adapter --limit 500 --train-log train_real.log
"""

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from grpo_reasoner.eval import evaluate

# dataviz palette (validated)
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASE_COLOR = "#2a78d6"   # slot 1 blue  (base model)
_GRPO_COLOR = "#eb6834"   # slot 2 orange (trained)


def parse_reward_curve(train_log: str) -> list[tuple[int, float]]:
    """Pull (step, correctness_reward_mean) points out of the TRL training log."""
    text = Path(train_log).read_text(encoding="utf-8", errors="ignore")
    points, step = [], 0
    for m in re.finditer(r"\{'loss'.*?\}", text):
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        step += 1
        val = d.get("rewards/correctness_reward/mean")
        if val is not None:
            points.append((step, float(val)))
    return points


def plot_before_after(base_acc: float, grpo_acc: float, n: int, out_path: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    labels = ["Base\n(Qwen2.5-0.5B)", "After GRPO"]
    vals = [base_acc, grpo_acc]
    bars = ax.bar(labels, vals, width=0.55, color=[_BASE_COLOR, _GRPO_COLOR], zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.1%}",
                ha="center", va="bottom", fontsize=12, color=_INK, fontweight="bold")
    delta = grpo_acc - base_acc
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_ylabel("GSM8K test accuracy (exact match)", fontsize=10, color=_INK2)
    ax.set_title(f"GRPO lifts GSM8K accuracy by {delta:+.1%}", fontsize=12.5, color=_INK, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=9.5, colors=_INK2)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.02, f"n = {n} held-out test problems, greedy decode", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=_MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=_SURFACE)
    plt.close(fig)


def plot_reward_curve(points: list[tuple[int, float]], out_path: str):
    if not points:
        print("No reward-curve points parsed from the training log (skipping curve).")
        return
    steps = [p[0] for p in points]
    rewards = [p[1] for p in points]
    # light smoothing for readability
    window = max(1, len(rewards) // 40)
    smoothed = [sum(rewards[max(0, i - window):i + 1]) / len(rewards[max(0, i - window):i + 1])
                for i in range(len(rewards))]

    fig, ax = plt.subplots(figsize=(7.0, 3.8), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.plot(steps, rewards, color=_GRPO_COLOR, alpha=0.25, linewidth=1, zorder=2)
    ax.plot(steps, smoothed, color=_GRPO_COLOR, linewidth=2, zorder=3)
    ax.set_xlabel("training step", fontsize=10, color=_INK2)
    ax.set_ylabel("mean correctness reward\n(fraction of rollouts correct)", fontsize=9.5, color=_INK2)
    ax.set_title("GRPO training: rollouts get more correct over time", fontsize=12, color=_INK, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.spines["left"].set_color("#c3c2b7")
    ax.tick_params(colors=_MUTED, labelsize=8.5)
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=_SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default="grpo-adapter")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-log", default="train_real.log")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating BASE {args.model} on {args.limit} test problems ...")
    base = evaluate(args.model, split="test", limit=args.limit, batch_size=args.batch_size,
                    save_path=str(out / "eval_base.jsonl"))
    print(f"  base accuracy = {base.accuracy:.1%}")

    print(f"Evaluating GRPO adapter {args.adapter} ...")
    grpo = evaluate(args.model, adapter_path=args.adapter, split="test", limit=args.limit,
                    batch_size=args.batch_size, save_path=str(out / "eval_grpo.jsonl"))
    print(f"  grpo accuracy = {grpo.accuracy:.1%}")

    plot_before_after(base.accuracy, grpo.accuracy, base.n, str(out / "before_after.png"))
    plot_reward_curve(parse_reward_curve(args.train_log), str(out / "reward_curve.png"))

    summary = {
        "model": args.model, "n": base.n,
        "base_accuracy": base.accuracy, "grpo_accuracy": grpo.accuracy,
        "delta": grpo.accuracy - base.accuracy,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
