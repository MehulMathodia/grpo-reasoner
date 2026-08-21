"""Gradio demo for a HuggingFace Space.

Type a math word problem and watch the GRPO-trained model reason step by step and
box its final answer. The Space loads the base model + the trained LoRA adapter
(both small enough for the free CPU/T4 Space tier).

Set MODEL_ID and ADAPTER_ID below to your pushed model, then this file + the
requirements next to it are all a HuggingFace Space needs.
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_ID = os.environ.get("ADAPTER_ID", "")  # e.g. "your-username/qwen2.5-0.5b-grpo-gsm8k"

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, showing your "
    "reasoning. Then give the final numeric answer on its own line in the form "
    "\\boxed{ANSWER}."
)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=_dtype).to(_device)
if ADAPTER_ID:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_ID).to(_device)
model.eval()


@torch.no_grad()
def solve(problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to(_device)
    out = model.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    completion = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return completion


EXAMPLES = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
]

title = "GRPO Math Reasoner"
description = (
    f"A **{MODEL_ID.split('/')[-1]}** model reinforcement-learning fine-tuned with **GRPO** "
    "(the DeepSeek-R1 algorithm) on GSM8K, using a verifiable correct-answer reward. "
    "Ask it a grade-school math word problem."
)

demo = gr.Interface(
    fn=solve,
    inputs=gr.Textbox(lines=3, label="Math word problem"),
    outputs=gr.Textbox(lines=12, label="Step-by-step solution"),
    examples=EXAMPLES,
    title=title,
    description=description,
)

if __name__ == "__main__":
    demo.launch()
