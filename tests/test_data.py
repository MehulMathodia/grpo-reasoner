from grpo_reasoner.data import extract_gold_answer, build_prompt, SYSTEM_PROMPT


def test_extract_gold_basic():
    answer = "He has 3 apples and buys 2 more.\n#### 5"
    assert extract_gold_answer(answer) == "5"


def test_extract_gold_strips_commas():
    answer = "The revenue works out to.\n#### 1,200"
    assert extract_gold_answer(answer) == "1200"


def test_extract_gold_negative():
    answer = "The net change is.\n#### -8"
    assert extract_gold_answer(answer) == "-8"


def test_extract_gold_raises_without_marker():
    try:
        extract_gold_answer("no marker here, just 42")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_prompt_shape():
    p = build_prompt("What is 2+2?")
    assert p[0]["role"] == "system"
    assert p[0]["content"] == SYSTEM_PROMPT
    assert p[1]["role"] == "user"
    assert p[1]["content"] == "What is 2+2?"
