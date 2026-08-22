"""GSM8K gold solutions -> SFT targets in the SAME output contract the GRPO
policy is rewarded for.

A raw GSM8K answer looks like:

    Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
    #### 72

For a fair GRPO-vs-SFT comparison the SFT model must be trained toward the
exact same target format the evaluator reads — so we strip the calculator
annotations (`<<...>>`, a GSM8K artifact the model should never emit) and turn
the `#### 72` line into `\boxed{72}`. Same system prompt, same chat format.
"""

import re

from datasets import load_dataset

from grpo_reasoner.data import build_prompt, extract_gold_answer

_CALC_RE = re.compile(r"<<[^>]*>>")
_FINAL_RE = re.compile(r"\n?####\s*-?[0-9][0-9,]*\s*$")


def convert_gold_solution(answer_field: str) -> str:
    """Strip <<calc>> annotations and replace the '#### N' line with \\boxed{N}."""
    gold = extract_gold_answer(answer_field)
    body = _CALC_RE.sub("", answer_field)
    body = _FINAL_RE.sub("", body).rstrip()
    return f"{body}\nThe final answer is \\boxed{{{gold}}}."


def load_gsm8k_sft(limit: int | None = None, seed: int = 0):
    """Prompt/completion conversational dataset for TRL's SFTTrainer.

    Columns: prompt (list of system+user messages), completion (list with the
    assistant message). `limit` takes the first N rows of a seeded shuffle, so
    the data-matched arm sees a random subset rather than the file's head."""
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    def _map(ex):
        return {
            "prompt": build_prompt(ex["question"]),
            "completion": [{"role": "assistant", "content": convert_gold_solution(ex["answer"])}],
        }

    return ds.map(_map, remove_columns=ds.column_names)
