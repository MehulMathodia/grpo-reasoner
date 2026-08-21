"""GRPO training: reinforcement-learning fine-tune a small LLM for GSM8K math.

This is the DeepSeek-R1-style recipe in miniature. GRPO (Group Relative Policy
Optimization) samples a GROUP of `num_generations` completions per question,
scores each with the verifiable reward functions, and pushes the policy toward
the above-average completions in each group — using the group mean as the
baseline, so no separate value/critic network is needed.

Design choices that matter here:
  * beta = 0.0 -> the KL-to-reference term is dropped and the reference model is
    never loaded. This halves memory (crucial on a 16GB T4) and follows the
    R1-Zero observation that reasoning can be incentivized without a KL anchor.
  * LoRA (peft) -> we train small adapters, not the full 1.5B, so it fits free
    compute and the artifact is a few-MB adapter.
  * dr_grpo loss -> the "Dr. GRPO" formulation, which removes the length bias
    present in the vanilla GRPO token-normalization.

Intended venue: a free Kaggle/Colab T4 (16GB). See notebooks/kaggle_train.md.
Local Windows + 8GB is fine for development and evaluation, but the full
rollout-heavy training run wants the T4's headroom and Linux vLLM support.
"""

import argparse

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from grpo_reasoner.data import load_gsm8k
from grpo_reasoner.rewards import REWARD_FUNCS


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO-train a small LLM on GSM8K.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output-dir", default="grpo-out")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-generations", type=int, default=8, help="Group size G (completions sampled per prompt).")
    p.add_argument("--per-device-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--train-limit", type=int, default=None, help="Cap training examples (default: full train split).")
    p.add_argument("--use-vllm", action="store_true", help="Use vLLM colocate generation (faster; needs vllm installed).")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.3, help="Fraction of GPU vLLM may reserve in colocate mode (leave room for training).")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=100)
    return p.parse_args()


def main():
    args = build_args()

    # T4 (Turing) has no bf16; Ampere+/Blackwell do. Pick the supported half-precision.
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # effective batch must be divisible by num_generations (GRPO constraint)
    effective = args.per_device_batch * args.grad_accum
    if effective % args.num_generations != 0:
        raise SystemExit(
            f"Effective batch {effective} (per_device {args.per_device_batch} x grad_accum "
            f"{args.grad_accum}) must be divisible by num_generations {args.num_generations}."
        )

    train_ds = load_gsm8k(split="train", limit=args.train_limit)
    # GRPOTrainer wants a `prompt` column of chat messages; keep gold for the reward
    train_ds = Dataset.from_dict({"prompt": train_ds["prompt"], "gold": train_ds["gold"]})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=0.0,                 # no reference model / KL term (R1-Zero style, saves memory)
        loss_type="dr_grpo",      # length-bias-corrected GRPO
        scale_rewards=True,
        gradient_checkpointing=True,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
        use_vllm=args.use_vllm,
        vllm_mode="colocate" if args.use_vllm else "server",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem if args.use_vllm else 0.3,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=REWARD_FUNCS,
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nSaved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
