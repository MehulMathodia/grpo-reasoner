"""Kaggle batch kernel: GRPO-train Qwen2.5-1.5B-Instruct on GSM8K, then evaluate
base vs trained on the held-out test split. Self-managing and GPU-adaptive:

  * detects the assigned GPU FIRST (Kaggle's API can hand out P100 or T4)
  * P100 (sm_60): reinstalls a torch build that still ships sm_60 kernels,
    skips vLLM (needs cc >= 7.5), uses a leaner training config
  * T4+ (sm_75+): keeps stock torch, uses vLLM for fast rollouts
  * every pip call is constrained so nothing can replace the working torch
  * training runs under a watchdog; checkpoints every 50 steps mean a timeout
    still yields a usable adapter; eval always gets its time slice
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

START = time.time()
TOTAL_BUDGET = 8.5 * 3600          # stay under Kaggle's ~9h batch wall
WORK = "/kaggle/working"
ADAPTER_DIR = f"{WORK}/grpo-adapter"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONSTRAINTS = f"{WORK}/constraints.txt"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def sh(cmd, timeout=None, check=True):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, timeout=timeout, check=check)


def pip(args, timeout=1800):
    sh([sys.executable, "-m", "pip", "install", "-q", "-c", CONSTRAINTS] + args, timeout=timeout)


def remaining():
    return TOTAL_BUDGET - (time.time() - START)


# ---------------------------------------------------------------- detect GPU
gpu_name = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip()
print(f"GPU: {gpu_name}", flush=True)
IS_P100 = "P100" in gpu_name

# ---------------------------------------------------------------- torch repair (P100)
if IS_P100:
    # stock image torch may lack sm_60 kernels; install a cu126 build that has them
    probe = subprocess.run(
        [sys.executable, "-c",
         "import torch; torch.zeros(2, device='cuda') @ torch.zeros(2,2, device='cuda')"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        print("stock torch cannot use the P100 — installing cu126 build", flush=True)
        sh([sys.executable, "-m", "pip", "install", "-q", "torch==2.13.0",
            "--index-url", "https://download.pytorch.org/whl/cu126"], timeout=2400)
        # the image's optional accelerator libs are built against the OLD torch
        # and poison imports (torchvision broke transformers, torchao broke
        # peft's LoRA dispatcher). Text-only LoRA GRPO needs none of them.
        sh([sys.executable, "-m", "pip", "uninstall", "-q", "-y",
            "torchvision", "torchaudio", "torchao", "bitsandbytes",
            "flash-attn", "xformers"], check=False)

# freeze whatever torch now works so later installs can't replace it
torch_ver = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True).stdout.strip()
with open(CONSTRAINTS, "w") as f:
    f.write(f"torch=={torch_ver}\n")
print(f"pinned torch=={torch_ver}", flush=True)

# ---------------------------------------------------------------- installs
pip(["trl==1.9.2", "peft==0.20.0", "transformers>=4.47", "datasets>=2.14", "accelerate>=0.30"])

VLLM_OK = False
if not IS_P100:
    try:
        pip(["vllm"])
        import importlib
        VLLM_OK = importlib.util.find_spec("vllm") is not None
        # verify torch survived the vllm install
        chk = subprocess.run([sys.executable, "-c",
                              "import torch; assert torch.cuda.is_available(); "
                              "torch.zeros(2, device='cuda') @ torch.zeros(2,2, device='cuda')"],
                             capture_output=True, text=True)
        if chk.returncode != 0:
            print("torch broken after vllm install — disabling vllm path", flush=True)
            VLLM_OK = False
    except Exception as e:  # noqa: BLE001
        print(f"vllm unavailable: {e}", flush=True)
print(f"VLLM_OK={VLLM_OK}", flush=True)

# ---------------------------------------------------------------- get the code
anchors = glob.glob("/kaggle/input/**/train_grpo.py", recursive=True)
if not anchors:
    for root, dirs, files in os.walk("/kaggle/input"):
        print(root, "->", files[:5], flush=True)
    raise AssertionError("code dataset not found under /kaggle/input")
src = os.path.dirname(anchors[0])
print(f"code found at: {src}", flush=True)
CODE = f"{WORK}/grpo-reasoner"
shutil.copytree(src, CODE, dirs_exist_ok=True)
pip(["-e", CODE])

# ---------------------------------------------------------------- train
def run_training(use_vllm: bool) -> bool:
    cmd = [sys.executable, f"{CODE}/train_grpo.py",
           "--model", MODEL,
           "--output-dir", ADAPTER_DIR,
           "--num-generations", "8",
           "--per-device-batch", "8",
           "--lr", "5e-6",
           "--save-steps", "50"]
    if use_vllm:
        cmd += ["--max-steps", "500", "--grad-accum", "4",
                "--max-completion-length", "512",
                "--use-vllm", "--vllm-gpu-mem", "0.25"]
    else:
        # plain HF generation is the bottleneck — leaner but still 2x the
        # local run's coverage and batch
        cmd += ["--max-steps", "300", "--grad-accum", "2",
                "--max-completion-length", "384"]
    budget = remaining() - 3.0 * 3600   # always reserve eval time
    try:
        sh(cmd, timeout=max(1800, budget))
        return True
    except subprocess.TimeoutExpired:
        print("training hit the watchdog — using last checkpoint", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"training crashed (exit {e.returncode})", flush=True)
        return False


ok = run_training(use_vllm=VLLM_OK)
if not ok and VLLM_OK:
    print("retrying WITHOUT vllm", flush=True)
    shutil.rmtree(ADAPTER_DIR, ignore_errors=True)
    ok = run_training(use_vllm=False)

adapter = None
if os.path.exists(f"{ADAPTER_DIR}/adapter_model.safetensors"):
    adapter = ADAPTER_DIR
else:
    ckpts = sorted(glob.glob(f"{ADAPTER_DIR}/checkpoint-*"),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    if ckpts:
        adapter = ckpts[-1]
assert adapter, "no adapter or checkpoint produced — training failed outright"
print(f"ADAPTER = {adapter}", flush=True)

# ---------------------------------------------------------------- evaluate
sys.path.insert(0, CODE)
from grpo_reasoner.eval import evaluate          # noqa: E402
from grpo_reasoner.majk import evaluate_majk     # noqa: E402

summary = {"model": MODEL, "gpu": gpu_name, "adapter": adapter, "vllm_used": VLLM_OK}

n_eval = 300 if remaining() > 2.5 * 3600 else 200
print(f"\n=== greedy eval (n={n_eval}): BASE ===", flush=True)
base = evaluate(MODEL, split="test", limit=n_eval, batch_size=16,
                save_path=f"{WORK}/eval_base_greedy.jsonl")
summary["base_greedy"] = base.accuracy
print(f"base greedy = {base.accuracy:.1%}", flush=True)

print(f"\n=== greedy eval (n={n_eval}): GRPO ===", flush=True)
grpo = evaluate(MODEL, adapter_path=adapter, split="test", limit=n_eval, batch_size=16,
                save_path=f"{WORK}/eval_grpo_greedy.jsonl")
summary["grpo_greedy"] = grpo.accuracy
summary["delta_greedy"] = grpo.accuracy - base.accuracy
summary["n_eval"] = n_eval
print(f"grpo greedy = {grpo.accuracy:.1%}  (delta {summary['delta_greedy']:+.1%})", flush=True)

if remaining() > 5400:
    print("\n=== maj@8 eval (bonus) ===", flush=True)
    try:
        bm = evaluate_majk(MODEL, split="test", limit=100, k=8, temperature=0.7, batch_size=32)
        gm = evaluate_majk(MODEL, adapter_path=adapter, split="test", limit=100, k=8,
                           temperature=0.7, batch_size=32)
        summary.update(base_majk=bm.majk_accuracy, grpo_majk=gm.majk_accuracy,
                       delta_majk=gm.majk_accuracy - bm.majk_accuracy,
                       base_pass=bm.mean_pass_rate, grpo_pass=gm.mean_pass_rate)
        print(f"maj@8: base {bm.majk_accuracy:.1%} -> grpo {gm.majk_accuracy:.1%}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"maj@8 skipped: {e}", flush=True)
else:
    print("skipping maj@8 (not enough time left)", flush=True)

with open(f"{WORK}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n=== FINAL SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)

shutil.rmtree(CODE, ignore_errors=True)
for ck in glob.glob(f"{ADAPTER_DIR}/checkpoint-*"):
    if os.path.abspath(ck) != os.path.abspath(adapter):
        shutil.rmtree(ck, ignore_errors=True)
