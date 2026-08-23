"""SFT baseline for the GRPO-vs-SFT ablation.

Same base model, same LoRA config (r=16, alpha=32, same target modules), same
output contract (\\boxed{}), loss on the assistant completion only. Two arms:

  --limit 600    data-matched: the same number of prompts the GRPO run saw
  --limit None   data-rich: all 7,473 gold solutions (~12x the prompts)

Learning rate is the standard SFT regime (1e-4 for LoRA), not GRPO's 5e-6 —
the two objectives live at different scales and each gets its own sensible
default; this is stated in the CHANGELOG rather than hidden.
"""

import argparse

import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from grpo_reasoner.sft_data import load_gsm8k_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", default="sft-adapter")
    p.add_argument("--limit", type=int, default=None, help="Number of training examples (None = all).")
    p.add_argument("--data-jsonl", default=None,
                   help="Train on a prebuilt prompt/completion JSONL (e.g. RFT data) instead of GSM8K gold.")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1, help="Override epochs if > 0.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--per-device-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=200)
    args = p.parse_args()

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.data_jsonl:
        from datasets import load_dataset
        train_ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    else:
        train_ds = load_gsm8k_sft(limit=args.limit)
    print(f"SFT examples: {len(train_ds)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        gradient_checkpointing=True,
        bf16=bf16_ok, fp16=not bf16_ok,
        logging_steps=10,
        save_steps=args.save_steps,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )
    trainer = SFTTrainer(
        model=args.model, args=config, train_dataset=train_ds,
        processing_class=tokenizer, peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nSaved SFT LoRA adapter to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
