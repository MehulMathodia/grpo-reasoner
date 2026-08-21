from grpo_reasoner.rewards import (
    extract_answer,
    correctness_reward,
    format_reward,
    _normalize_number,
)


# --- answer extraction ---

def test_extract_from_boxed():
    assert extract_answer("The total is \\boxed{42} dollars.") == "42"


def test_extract_prefers_last_boxed():
    assert extract_answer("First \\boxed{7}, correction \\boxed{12}.") == "12"


def test_extract_falls_back_to_last_number():
    # no box, but a final number is present
    assert extract_answer("Adding them gives 5 and 3, so the answer is 8") == "8"


def test_extract_strips_commas():
    assert extract_answer("\\boxed{1,234}") == "1234"


def test_extract_normalizes_decimal():
    assert extract_answer("\\boxed{42.0}") == "42"


def test_extract_none_when_no_number():
    assert extract_answer("I am not sure how to solve this.") is None


def test_normalize_number():
    assert _normalize_number("1,000") == "1000"
    assert _normalize_number("3.50") == "3.5"
    assert _normalize_number("7.0") == "7"


# --- correctness reward ---

def test_correctness_reward_hits_and_misses():
    completions = ["\\boxed{42}", "the answer is 41", "\\boxed{1,000}"]
    gold = ["42", "42", "1000"]
    assert correctness_reward(completions, gold) == [1.0, 0.0, 1.0]


def test_correctness_reward_accepts_chat_format_completion():
    # TRL may pass completions as chat lists
    completions = [[{"role": "assistant", "content": "so \\boxed{9}"}]]
    gold = ["9"]
    assert correctness_reward(completions, gold) == [1.0]


def test_correctness_reward_wrong_but_boxed_is_zero():
    # being well-formatted but wrong must score 0 on correctness
    assert correctness_reward(["\\boxed{99}"], ["42"]) == [0.0]


# --- format reward ---

def test_format_reward_rewards_box_only():
    completions = ["\\boxed{42}", "the answer is 42", "no answer here"]
    assert format_reward(completions) == [0.2, 0.0, 0.0]


def test_format_reward_strictly_less_than_correct():
    # a wrong-but-boxed answer (format 0.2) must never out-earn a correct answer
    # (correctness 1.0) — the shaping reward can't dominate the objective
    boxed_wrong_total = format_reward(["\\boxed{99}"])[0] + correctness_reward(["\\boxed{99}"], ["42"])[0]
    correct_total = format_reward(["\\boxed{42}"])[0] + correctness_reward(["\\boxed{42}"], ["42"])[0]
    assert correct_total > boxed_wrong_total
