# How the 1.5B run was actually produced (Kaggle, free tier)

This documents what really happened, not an idealized recipe.

The run was driven entirely from a local terminal via the Kaggle API
(`kaggle kernels push`), not from the notebook UI. The two scripts in this folder
are the exact kernels that produced the reported numbers:

- **`kaggle_train_kernel.py`** — environment repair + GRPO training + n=300 eval
  (kernel `grpo-reasoner-train-1-5b`, version 5; ~5.3 h)
- **`kaggle_eval_kernel.py`** — full 1,319-question greedy eval of base vs adapter,
  with the paired McNemar test computed in-kernel (kernel `grpo-reasoner-eval-full`;
  ~5 h). It mounts the training kernel's output as input, so the adapter is never
  re-uploaded.

## What the environment fight looked like (versions 1-4 all failed, each < 15 min)

API-pushed kernels on this account were always assigned a **Tesla P100 (sm_60)**,
not a T4 — and there is no accelerator-selection field in the kernel-metadata API.
That drove every fix:

1. v1 — wrong assumption about where Kaggle mounts the code dataset → recursive search.
2. v2 — `pip install vllm` replaced torch with a cu130 build that has **no sm_60
   kernels**; torch died on the P100 → detect GPU first, never install vLLM on P100.
3. v3 — the stock torch itself lacked sm_60 support; reinstalling `torch==2.13.0`
   from the **cu126** index fixed torch but the image's torchvision (built for the
   old torch) then broke transformers' import chain.
4. v4 — torchvision removed; then the image's stale **torchao 0.10** broke peft's
   LoRA dispatcher.
5. v5 — removed the whole family of stale accelerator libs (torchvision, torchaudio,
   torchao, bitsandbytes, flash-attn, xformers) and pinned torch via a pip
   constraints file so nothing could replace it again. **Ran to completion.**

So: **vLLM was NOT used** for the reported run. Generation was plain HF
`model.generate` on a P100, which is why the config is 300 steps × effective batch
16 (2 prompt-groups/step) rather than the 500 × 32 that a T4+vLLM would allow.

## Exact training config (v5, as run)

```
Qwen/Qwen2.5-1.5B-Instruct · LoRA r=16 α=32 on all attention+MLP projections
GRPO: num_generations=8, per_device_batch=8, grad_accum=2 (effective 16)
max_steps=300, max_completion_length=384, lr=5e-6, beta=0.0, loss=dr_grpo, fp16
```

## Reproduce it yourself

```bash
pip install kaggle            # + a Kaggle API token in ~/.kaggle/
# upload the code as a private dataset, then push the kernels:
kaggle datasets create -p <dir-with-dataset-metadata.json> --dir-mode zip
kaggle kernels push -p <dir-with-kernel-metadata.json>
kaggle kernels status <user>/grpo-reasoner-train-1-5b
kaggle kernels output <user>/grpo-reasoner-train-1-5b -p out/
```

If your kernel lands on a T4 instead, the train kernel will use vLLM and the
larger config automatically (that branch is written but was never exercised here).
