"""Transfer-eval Kaggle kernel (v0.2): does the GSM8K-trained GRPO adapter help on
benchmarks it never saw?

  * SVAMP (all 1,000) — different source/distribution
  * GSM-Hard (as many as time allows, up to 1,319) — same questions, hard numbers

Base vs adapter, greedy, paired McNemar per benchmark. Adapter mounted from the
training kernel's output (kernel_sources); code from the dataset (v2).
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


def mcnemar(base_recs, grpo_recs, key="question"):
    b_map = {r[key]: r["correct"] for r in base_recs}
    g_map = {r[key]: r["correct"] for r in grpo_recs}
    b = sum(1 for q in b_map if b_map[q] and not g_map.get(q, False))
    c = sum(1 for q in b_map if not b_map[q] and g_map.get(q, False))
    n = b + c
    p = min(1.0, sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n * 2) if n else 1.0
    return {"regressions": b, "fixes": c, "mcnemar_p": p}


# ---------------- GPU + torch repair (identical to the runs that worked)
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
    "transformers>=4.47", "datasets>=2.14", "peft==0.20.0", "accelerate>=0.30"], timeout=1200)

# ---------------- inputs
anchors = glob.glob("/kaggle/input/**/grpo_reasoner/transfer.py", recursive=True)
adapters = glob.glob("/kaggle/input/**/grpo-adapter/adapter_model.safetensors", recursive=True)
if not anchors or not adapters:
    for root, dirs, files in os.walk("/kaggle/input"):
        print(root, "->", files[:6], flush=True)
    raise AssertionError(f"missing inputs: code={bool(anchors)} adapter={bool(adapters)}")
CODE = os.path.dirname(os.path.dirname(anchors[0]))
ADAPTER = f"{WORK}/grpo-adapter"
shutil.copytree(os.path.dirname(adapters[0]), ADAPTER, dirs_exist_ok=True)
sys.path.insert(0, CODE)
print(f"code: {CODE}\nadapter: {ADAPTER}", flush=True)

from grpo_reasoner.eval import evaluate_items                 # noqa: E402
from grpo_reasoner.transfer import load_svamp, load_gsm_hard, answers_match  # noqa: E402

summary = {"model": MODEL, "gpu": gpu_name, "benchmarks": {}}


def run_benchmark(name, items):
    t0 = time.time()
    print(f"\n=== {name}: BASE (n={len(items)}) ===", flush=True)
    base = evaluate_items(MODEL, items, batch_size=16, matcher=answers_match,
                          save_path=f"{WORK}/{name}_base.jsonl")
    print(f"  base = {base.accuracy:.2%}", flush=True)
    print(f"=== {name}: GRPO ===", flush=True)
    grpo = evaluate_items(MODEL, items, adapter_path=ADAPTER, batch_size=16, matcher=answers_match,
                          save_path=f"{WORK}/{name}_grpo.jsonl")
    print(f"  grpo = {grpo.accuracy:.2%}  (delta {grpo.accuracy - base.accuracy:+.2%})", flush=True)
    b_recs = [json.loads(l) for l in open(f"{WORK}/{name}_base.jsonl", encoding="utf-8")]
    g_recs = [json.loads(l) for l in open(f"{WORK}/{name}_grpo.jsonl", encoding="utf-8")]
    stats = mcnemar(b_recs, g_recs)
    summary["benchmarks"][name] = {
        "n": base.n, "base": base.accuracy, "grpo": grpo.accuracy,
        "delta": grpo.accuracy - base.accuracy, **stats,
        "seconds": time.time() - t0,
    }
    print(json.dumps(summary["benchmarks"][name], indent=2), flush=True)
    with open(f"{WORK}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return time.time() - t0


# SVAMP first (all 1,000), then size GSM-Hard by the measured rate
sv_items = load_svamp()
sv_seconds = run_benchmark("svamp", sv_items)
per_item = sv_seconds / (2 * len(sv_items))          # seconds per question per model
budget = remaining() - 900                           # keep a 15-min tail
n_hard = int(min(1319, budget / (2 * per_item * 1.15)))  # GSM-Hard answers run longer; 15% pad
n_hard = max(0, (n_hard // 50) * 50)
print(f"\nSVAMP took {sv_seconds/60:.1f} min ({per_item:.2f}s/item/model); "
      f"GSM-Hard n = {n_hard}", flush=True)
if n_hard >= 200:
    run_benchmark("gsm_hard", load_gsm_hard(limit=n_hard))
else:
    print("not enough time left for a meaningful GSM-Hard sample; skipped", flush=True)

summary["elapsed_hours"] = (time.time() - START) / 3600
with open(f"{WORK}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n=== FINAL SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
