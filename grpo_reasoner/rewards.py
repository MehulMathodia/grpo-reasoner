"""Verifiable reward functions for GRPO on GSM8K.

These are the heart of the project. GRPO improves the policy by sampling a group
of completions per question and pushing probability toward the ones that score
higher, so the reward design *is* the training signal — a sloppy extractor or a
reward that's gameable will train a worse model regardless of the RL algorithm.

Two rewards, deliberately separated:

  * correctness_reward — the real objective: does the extracted final answer
    equal the gold number? Verifiable, no judge model.
  * format_reward — a small shaping reward for actually using the \boxed{}
    contract, so early in training (when the model is rarely correct and the
    correctness signal is sparse) there is still gradient toward producing a
    parseable answer.

Each function has the signature TRL's GRPOTrainer expects: it receives the list
of `completions` plus any dataset columns as keyword arguments (here `gold`),
and returns one float per completion.
"""

import re

_BOXED_RE = re.compile(r"\\boxed\{\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*\}")
_NUMBER_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def _completion_text(completion) -> str:
    """TRL passes each completion either as a raw string or as a chat list
    [{'role': 'assistant', 'content': ...}]. Normalize to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


def _normalize_number(s: str) -> str:
    """Canonicalize a numeric string for exact comparison: drop thousands commas,
    strip a trailing '.0' so '42' and '42.0' match."""
    s = s.replace(",", "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def extract_answer(text: str) -> str | None:
    """Extract the model's final numeric answer.

    Priority: the LAST \boxed{...} (the contract), falling back to the LAST bare
    number in the text (models often reason to the right answer but forget the
    box — we still want to reward a correct final number, and picking the last
    number matches where a final answer normally appears)."""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return _normalize_number(boxed[-1])
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return _normalize_number(numbers[-1])
    return None


def correctness_reward(completions, gold, **kwargs) -> list[float]:
    """1.0 if the extracted final answer equals the gold number, else 0.0."""
    rewards = []
    for completion, gold_answer in zip(completions, gold):
        pred = extract_answer(_completion_text(completion))
        rewards.append(1.0 if pred is not None and pred == _normalize_number(gold_answer) else 0.0)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """0.2 if the completion contains a well-formed \boxed{number}, else 0.0.

    Small relative to correctness (1.0) so it shapes behavior without letting the
    model farm reward by boxing a wrong answer — being correct is always worth
    strictly more than merely being well-formatted."""
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        rewards.append(0.2 if _BOXED_RE.search(text) else 0.0)
    return rewards


# Convenience: the reward set the training script passes to GRPOTrainer.
REWARD_FUNCS = [correctness_reward, format_reward]
