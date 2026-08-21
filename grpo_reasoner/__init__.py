"""grpo-reasoner — GRPO reinforcement-learning fine-tuning of a small LLM for math reasoning."""

from grpo_reasoner.rewards import correctness_reward, format_reward, extract_answer, REWARD_FUNCS
from grpo_reasoner.data import load_gsm8k, extract_gold_answer, SYSTEM_PROMPT

__version__ = "0.1.0"

__all__ = [
    "correctness_reward",
    "format_reward",
    "extract_answer",
    "REWARD_FUNCS",
    "load_gsm8k",
    "extract_gold_answer",
    "SYSTEM_PROMPT",
]
