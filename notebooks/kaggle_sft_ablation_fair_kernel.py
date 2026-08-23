"""GRPO-vs-SFT ablation, fairness pass (v0.3, part 2).

The first pass (lr 1e-4 on gold rationales) collapsed completions to ~235 chars
(base: ~956) and dropped accuracy ~20 pts below base. Before calling that a
finding, run the baselines a skeptic would demand:

  * sft600_lr2e-5  / sftfull_lr2e-5 — gold rationales at a gentle LR
  * rft600_lr2e-5  / rft600_lr1e-4  — rejection-sampling FT: sample from the
    BASE model on the same 600 prompts, keep only verified-correct solutions,
    SFT on those (on-policy; the strongest fair SFT baseline)

Each arm: same base, same LoRA, same output contract, full GSM8K test, paired
McNemar vs base and vs the GRPO adapter.
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
    a = {r["question"]: r["correct"] for r in a_recs}
    b = {r["question"]: r["correct"] for r in b_recs}
    common = [q for q in a if q in b]
    reg = sum(1 for q in common if a[q] and not b[q])
    fix = sum(1 for q in common if not a[q] and b[q])
    n = reg + fix
    p = min(1.0, sum(comb(n, k) for k in range(0, min(reg, fix) + 1)) / 2**n * 2) if n else 1.0
    return {"n_common": len(common), "regressions": reg, "fixes": fix, "mcnemar_p": p}


# ---------------- env repair
gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
print(f"GPU: {gpu_name}", flush=True)
if "P100" in gpu_name:
    probe = subprocess.run([sys.executable, "-c",
                            "import torch; torch.zeros(2, device='cuda') @ torch.zeros(2,2, device='cuda')"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
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
anchors = glob.glob("/kaggle/input/**/grpo_reasoner/rft.py", recursive=True)
refs = glob.glob("/kaggle/input/**/reference/eval_base_full.jsonl", recursive=True)
if not anchors or not refs:
    for root, dirs, files in os.walk("/kaggle/input"):
        print(root, "->", files[:6], flush=True)
    raise AssertionError(f"missing inputs: code={bool(anchors)} reference={bool(refs)}")
CODE = f"{WORK}/grpo-reasoner"
shutil.copytree(os.path.dirname(os.path.dirname(anchors[0])), CODE, dirs_exist_ok=True)
sys.path.insert(0, CODE)
REF = f"{CODE}/reference"

from grpo_reasoner.eval import evaluate          # noqa: E402
from grpo_reasoner.rft import build_rft_dataset  # noqa: E402

base_recs = load_jsonl(f"{REF}/eval_base_full.jsonl")
grpo_recs = load_jsonl(f"{REF}/eval_grpo_full.jsonl")
summary = {"model": MODEL, "gpu": gpu_name, "arms": {}, "train_seconds": {}}


def train_sft(name, lr, epochs, limit=None, data_jsonl=None):
    out = f"{WORK}/{name}"
    cmd = [sys.executable, f"{CODE}/train_sft.py", "--model", MODEL, "--output-dir", out,
           "--epochs", str(epochs), "--per-device-batch", "8", "--grad-accum", "2",
           "--max-length", "768", "--lr", str(lr), "--save-steps", "100000"]
    if data_jsonl:
        cmd += ["--data-jsonl", data_jsonl]
    elif limit:
        cmd += ["--limit", str(limit)]
    t0 = time.time()
    sh(cmd, timeout=2.5 * 3600)
    for ck in glob.glob(f"{out}/checkpoint-*"):
        shutil.rmtree(ck, ignore_errors=True)
    summary["train_seconds"][name] = time.time() - t0
    print(f"{name}: trained in {(time.time()-t0)/60:.1f} min", flush=True)
    return out


# ---------------- RFT data from the BASE model (same 600 prompts as the gold arm)
t0 = time.time()
rft_path = f"{WORK}/rft600_data.jsonl"
prebuilt = glob.glob("/kaggle/input/**/reference/rft600_data.jsonl", recursive=True)
if prebuilt:
    shutil.copy(prebuilt[0], rft_path)
    shutil.copy(prebuilt[0].replace(".jsonl", "_stats.json"),
                rft_path.replace(".jsonl", "_stats.json"))
    print("using prebuilt RFT data from the previous run (generation already verified)", flush=True)
else:
    build_rft_dataset(MODEL, n_prompts=600, k=4, temperature=0.7, max_per_prompt=2,
                      max_new_tokens=512, batch_size=32, seed=0, out_path=rft_path)
summary["rft_gen_seconds"] = time.time() - t0
summary["rft_stats"] = json.load(open(rft_path.replace(".jsonl", "_stats.json")))

# ---------------- train all arms (cheap), then evaluate in priority order
arms = {
    "rft600_lr1e-4":   train_sft("rft600_lr1e-4", lr=1e-4, epochs=3, data_jsonl=rft_path),
    "sft600_lr2e-5":   train_sft("sft600_lr2e-5", lr=2e-5, epochs=3, limit=600),
    "sftfull_lr2e-5":  train_sft("sftfull_lr2e-5", lr=2e-5, epochs=1, limit=None),
    "rft600_lr2e-5":   train_sft("rft600_lr2e-5", lr=2e-5, epochs=3, data_jsonl=rft_path),
}


def eval_arm(name, adapter, n):
    t0 = time.time()
    print(f"\n=== eval {name} (n={n}) ===", flush=True)
    res = evaluate(MODEL, adapter_path=adapter, split="test", limit=n, batch_size=16,
                   save_path=f"{WORK}/eval_{name}.jsonl")
    recs = load_jsonl(f"{WORK}/eval_{name}.jsonl")
    lens = sorted(len(r["completion"]) for r in recs)
    summary["arms"][name] = {
        "n": res.n, "accuracy": res.accuracy, "median_completion_chars": lens[len(lens)//2],
        "vs_base": mcnemar(base_recs, recs), "vs_grpo": mcnemar(grpo_recs, recs),
        "eval_seconds": time.time() - t0,
    }
    print(json.dumps(summary["arms"][name], indent=2), flush=True)
    with open(f"{WORK}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return time.time() - t0


per_item = None
for name in ["rft600_lr1e-4", "sft600_lr2e-5", "sftfull_lr2e-5", "rft600_lr2e-5"]:
    if per_item is None:
        n = 1319
    else:
        n = int(min(1319, (remaining() - 900) / (per_item * 1.1)))
        n = max(0, (n // 50) * 50)
    if n < 300:
        print(f"skipping {name}: not enough time (n={n})", flush=True)
        continue
    secs = eval_arm(name, arms[name], n)
    per_item = secs / n

summary["elapsed_hours"] = (time.time() - START) / 3600
with open(f"{WORK}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n=== FINAL SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
shutil.rmtree(CODE, ignore_errors=True)
