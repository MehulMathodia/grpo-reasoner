"""Shared inference: load a (optionally LoRA-adapted) model and batch-generate.

Used by both the eval harness and the demo, so that what we measure and what we
ship behave identically."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, adapter_path: str | None = None, device: str | None = None):
    """Load a causal LM and tokenizer. If `adapter_path` is given, load that LoRA
    adapter on top of the base `model_id` (this is how we evaluate the GRPO-trained
    policy without merging weights)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).to(device)

    model.eval()
    # left-padding is required for correct batched generation with decoder-only LMs
    tokenizer.padding_side = "left"
    return model, tokenizer


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: list[list[dict]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 16,
) -> list[str]:
    """Generate completions for a list of chat-format prompts. Greedy by default
    (temperature=0) for deterministic, reproducible evaluation."""
    device = next(model.parameters()).device
    outputs = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        gen = model.generate(**enc, **gen_kwargs)
        # strip the prompt tokens; keep only newly generated tokens
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    return outputs
