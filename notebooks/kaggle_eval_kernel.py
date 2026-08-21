"""Eval-only Kaggle kernel: greedy exact-match on the FULL GSM8K test split
(1,319 questions), base Qwen2.5-1.5B-Instruct vs the GRPO adapter produced by
the training kernel (mounted as input via kernel_sources).

No training here — this exists purely to decide, with a decisive sample size,
whether the +2.7-point gain measured at n=300 is real.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

START = time.time()
WORK = "/kaggle/working"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONSTRAINTS = f"{WORK}/constraints.txt"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def sh(cmd, timeout=None, check=True):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, timeout=timeout, check=check)


# ---------------- GPU + torch repair (same battle-tested sequence as training)
gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
print(f"GPU: {gpu_name}", flush=True)
if "P100" in gpu_name:
    probe = subprocess.run(
        [sys.executable, "-c",
         "import torch; torch.zeros(2, device='cuda') @ torch.zeros(2,2, device='cuda')"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        print("repairing torch for P100 (cu126) + removing stale accelerator libs", flush=True)
        sh([sys.executable, "-m", "pip", "install", "-q", "torch==2.13.0",
            "--index-url", "https://download.pytorch.org/whl/cu126"], timeout=2400)
        sh([sys.executable, "-m", "pip", "uninstall", "-q", "-y",
            "torchvision", "torchaudio", "torchao", "bitsandbytes",
            "flash-attn", "xformers"], check=False)

torch_ver = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                           capture_output=True, text=True).stdout.strip()
with open(CONSTRAINTS, "w") as f:
    f.write(f"torch=={torch_ver}\n")
sh([sys.executable, "-m", "pip", "install", "-q", "-c", CONSTRAINTS,
    "transformers>=4.47", "datasets>=2.14", "peft==0.20.0", "accelerate>=0.30"], timeout=1200)

# ---------------- locate code + adapter among the mounted inputs
anchors = glob.glob("/kaggle/input/**/grpo_reasoner/eval.py", recursive=True)
adapters = glob.glob("/kaggle/input/**/grpo-adapter/adapter_model.safetensors", recursive=True)
if not anchors or not adapters:
    for root, dirs, files in os.walk("/kaggle/input"):
        print(root, "->", files[:6], flush=True)
    raise AssertionError(f"missing inputs: code={bool(anchors)} adapter={bool(adapters)}")
CODE = os.path.dirname(os.path.dirname(anchors[0]))
ADAPTER_SRC = os.path.dirname(adapters[0])
# peft may want to write nothing, but keep the adapter on a writable path anyway
ADAPTER = f"{WORK}/grpo-adapter"
shutil.copytree(ADAPTER_SRC, ADAPTER, dirs_exist_ok=True)
print(f"code: {CODE}\nadapter: {ADAPTER}", flush=True)
sys.path.insert(0, CODE)

from grpo_reasoner.eval import evaluate  # noqa: E402

# ---------------- full-test evaluation, base then adapter
summary = {"model": MODEL, "gpu": gpu_name, "n_eval": "full-test-1319"}

print("\n=== FULL greedy eval: BASE ===", flush=True)
base = evaluate(MODEL, split="test", limit=None, batch_size=16,
                save_path=f"{WORK}/eval_base_full.jsonl")
summary["base_greedy"] = base.accuracy
summary["n"] = base.n
print(f"base = {base.accuracy:.2%} on {base.n}", flush=True)

print("\n=== FULL greedy eval: GRPO ===", flush=True)
grpo = evaluate(MODEL, adapter_path=ADAPTER, split="test", limit=None, batch_size=16,
                save_path=f"{WORK}/eval_grpo_full.jsonl")
summary["grpo_greedy"] = grpo.accuracy
summary["delta_greedy"] = grpo.accuracy - base.accuracy
print(f"grpo = {grpo.accuracy:.2%}  (delta {summary['delta_greedy']:+.2%})", flush=True)

# ---------------- paired McNemar, computed in-kernel
from math import comb  # noqa: E402
bmap, gmap = {}, {}
for line in open(f"{WORK}/eval_base_full.jsonl", encoding="utf-8"):
    r = json.loads(line); bmap[r["question"]] = r["correct"]
for line in open(f"{WORK}/eval_grpo_full.jsonl", encoding="utf-8"):
    r = json.loads(line); gmap[r["question"]] = r["correct"]
b = sum(1 for q in bmap if bmap[q] and not gmap.get(q, False))
c = sum(1 for q in bmap if not bmap[q] and gmap.get(q, False))
n_disc = b + c
p = min(1.0, sum(comb(n_disc, k) for k in range(0, min(b, c) + 1)) / 2**n_disc * 2) if n_disc else 1.0
summary.update(regressions=b, fixes=c, mcnemar_p=p)

with open(f"{WORK}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n=== FINAL SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
print(f"\nelapsed: {(time.time()-START)/3600:.2f} h", flush=True)
