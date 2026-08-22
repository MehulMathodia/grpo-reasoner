from grpo_reasoner.sft_data import convert_gold_solution
from grpo_reasoner.rewards import extract_answer


RAW = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)


def test_strips_calculator_annotations():
    out = convert_gold_solution(RAW)
    assert "<<" not in out and ">>" not in out
    assert "48/2 = 24 clips" in out


def test_replaces_final_marker_with_boxed():
    out = convert_gold_solution(RAW)
    assert "####" not in out
    assert out.rstrip().endswith("\\boxed{72}.")


def test_same_extractor_reads_the_converted_target():
    # the SFT target must be gradeable by the exact evaluator used for GRPO
    out = convert_gold_solution(RAW)
    assert extract_answer(out) == "72"


def test_handles_comma_and_negative_gold():
    raw = "Revenue was 1,200 and costs 1,500.\n#### -300"
    out = convert_gold_solution(raw)
    assert out.endswith("\\boxed{-300}.")
    assert extract_answer(out) == "-300"
    raw2 = "Total sales.\n#### 1,200"
    assert extract_answer(convert_gold_solution(raw2)) == "1200"
