"""GRPO-vs-SFT ablation kernel (v0.3).

Trains two SFT LoRA baselines on GSM8K gold solutions (same base, same LoRA
config, same \\boxed{} output contract, completion-only loss):
  * sft600  — data-matched: 600 prompts, the number the GRPO run saw
  * sftfull — data-rich: all 7,473 gold solutions (~12x)
then evaluates each on the full 1,319-question GSM8K test split and runs paired
McNemar against BOTH the base model and the GRPO adapter (reference per-question
files shipped in the code dataset, produced by the earlier eval kernel).
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from math import comb

START = time.time()
TOTAL_BUDGET = 8.5 * 3600
WORK = "/kaggle/working"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONSTRAINTS = f"{WORK}/constraints.txt"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def sh(cmd, timeout=None, check=True):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, timeout=timeout, check=check)


def remaining():
    return TOTAL_BUDGET - (time.time() - START)


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def mcnemar(a_recs, b_recs):
    """Paired test of B vs A on shared questions: fixes = A wrong & B right."""
    a = {r["question"]: r["correct"] for r in a_recs}
    b = {r["question"]: r["correct"] for r in b_recs}
    common = [q for q in a if q in b]
    reg = sum(1 for q in common if a[q] and not b[q])
    fix = sum(1 for q in common if not a[q] and b[q])
    n = reg + fix
    p = min(1.0, sum(comb(n, k) for k in range(0, min(reg, fix) + 1)) / 2**n * 2) if n else 1.0
    return {"n_common": len(common), "regressions": reg, "fixes": fix, "mcnemar_p": p}


# ---------------- env repair (identical to the runs that worked)
gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
print(f"GPU: {gpu_name}", flush=True)
if "P100" in gpu_name:
    probe = subprocess.run([sys.executable, "-c",
                            "import torch; torch.zeros(2, device='cuda') @ torch.zeros(2,2, device='cuda')"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        print("repairing torch for P100 (cu126) + removing stale accelerator libs", flush=True)
        sh([sys.executable, "-m", "pip", "install", "-q", "torch==2.13.0",
            "--index-url", "https://download.pytorch.org/whl/cu126"], timeout=2400)
        sh([sys.executable, "-m", "pip", "uninstall", "-q", "-y",
            "torchvision", "torchaudio", "torchao", "bitsandbytes", "flash-attn", "xformers"], check=False)
torch_ver = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                           capture_output=True, text=True).stdout.strip()
with open(CONSTRAINTS, "w") as f:
    f.write(f"torch=={torch_ver}\n")
sh([sys.executable, "-m", "pip", "install", "-q", "-c", CONSTRAINTS,
    "trl==1.9.2", "peft==0.20.0", "transformers>=4.47", "datasets>=2.14", "accelerate>=0.30"], timeout=1500)

# ---------------- inputs
anchors = glob.glob("/kaggle/input/**/train_sft.py", recursive=True)
refs = glob.glob("/kaggle/input/**/reference/eval_base_full.jsonl", recursive=True)
if not anchors or not refs:
    for root, dirs, files in os.walk("/kaggle/input"):
        print(root, "->", files[:6], flush=True)
    raise AssertionError(f"missing inputs: code={bool(anchors)} reference={bool(refs)}")
CODE = f"{WORK}/grpo-reasoner"
shutil.copytree(os.path.dirname(anchors[0]), CODE, dirs_exist_ok=True)
sys.path.insert(0, CODE)
REF = f"{CODE}/reference"
print(f"code: {CODE}", flush=True)

from grpo_reasoner.eval import evaluate  # noqa: E402

summary = {"model": MODEL, "gpu": gpu_name, "arms": {}}


def train_sft(name, limit, epochs):
    out = f"{WORK}/{name}"
    cmd = [sys.executable, f"{CODE}/train_sft.py", "--model", MODEL, "--output-dir", out,
           "--epochs", str(epochs), "--per-device-batch", "8", "--grad-accum", "2",
           "--max-length", "768", "--lr", "1e-4", "--save-steps", "1000"]
    if limit:
        cmd += ["--limit", str(limit)]
    t0 = time.time()
    sh(cmd, timeout=2.5 * 3600)
    secs = time.time() - t0
    # drop optimizer checkpoints to keep output small
    for ck in glob.glob(f"{out}/checkpoint-*"):
        shutil.rmtree(ck, ignore_errors=True)
    print(f"{name}: trained in {secs/60:.1f} min", flush=True)
    return out, secs


base_recs = load_jsonl(f"{REF}/eval_base_full.jsonl")
grpo_recs = load_jsonl(f"{REF}/eval_grpo_full.jsonl")
summary["reference"] = {"base_acc": sum(r["correct"] for r in base_recs) / len(base_recs),
                        "grpo_acc": sum(r["correct"] for r in grpo_recs) / len(grpo_recs),
                        "grpo_vs_base": mcnemar(base_recs, grpo_recs)}

# ---------------- train both arms first (cheap), then evaluate
sft600_dir, t600 = train_sft("sft600", limit=600, epochs=3)
sftfull_dir, tfull = train_sft("sftfull", limit=None, epochs=1)
summary["train_seconds"] = {"sft600": t600, "sftfull": tfull}


def eval_arm(name, adapter, n):
    t0 = time.time()
    print(f"\n=== eval {name} on GSM8K test (n={n}) ===", flush=True)
    res = evaluate(MODEL, adapter_path=adapter, split="test", limit=n, batch_size=16,
                   save_path=f"{WORK}/eval_{name}.jsonl")
    recs = load_jsonl(f"{WORK}/eval_{name}.jsonl")
    summary["arms"][name] = {
        "n": res.n, "accuracy": res.accuracy,
        "vs_base": mcnemar(base_recs, recs),
        "vs_grpo": mcnemar(grpo_recs, recs),
        "eval_seconds": time.time() - t0,
    }
    print(json.dumps(summary["arms"][name], indent=2), flush=True)
    with open(f"{WORK}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return time.time() - t0


secs = eval_arm("sft600", sft600_dir, 1319)
per_item = secs / 1319
n_full = int(min(1319, (remaining() - 900) / (per_item * 1.1)))
n_full = max(0, (n_full // 50) * 50)
print(f"\nsft600 eval {secs/60:.1f} min; sftfull eval n = {n_full}", flush=True)
if n_full >= 300:
    eval_arm("sftfull", sftfull_dir, n_full)
else:
    print("not enough time for sftfull eval", flush=True)

summary["elapsed_hours"] = (time.time() - START) / 3600
with open(f"{WORK}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n=== FINAL SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
shutil.rmtree(CODE, ignore_errors=True)
