from grpo_reasoner.transfer import answers_match, grade


def test_integer_vs_float_formatting():
    assert answers_match("42", "42.0")
    assert answers_match("42.00", 42)
    assert answers_match("27", "27")


def test_commas_and_negatives():
    assert answers_match("1,234", 1234.0)
    assert answers_match("-9,867,630", "-9867630.0")
    assert answers_match("-5", -5.0)


def test_decimal_answers():
    assert answers_match("3.5", "3.50")
    assert answers_match("0.25", 0.25)


def test_mismatches_are_rejected():
    assert not answers_match("42", "43")
    assert not answers_match("1000", "100")
    assert not answers_match("3.5", "3.4")
    assert not answers_match(None, "42")
    assert not answers_match("forty-two", "42")


def test_tolerance_is_tight():
    # 1e-4 relative: 10000 vs 10001 must NOT match (rel diff 1e-4 exactly at edge -> isclose says True at <=)
    assert not answers_match("10000", "10003")
    assert not answers_match("99", "100")


def test_grade_extracts_then_matches():
    pred, ok = grade("Total is \\boxed{-9,867,630}.", "-9867630.0")
    assert pred == "-9867630" and ok
    pred, ok = grade("so the answer is 27", "27.0")
    assert pred == "27" and ok
    pred, ok = grade("I cannot solve this.", "27")
    assert pred is None and not ok
