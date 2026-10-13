import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "verl"
    / "utils"
    / "reward_score"
    / "pair_rm_with_tool_reward_v8.py"
)
_SPEC = importlib.util.spec_from_file_location("pair_rm_with_tool_reward_v8", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
reward_v8 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reward_v8)


def _final_answer(choice: int) -> str:
    return (
        "assistant\n"
        "<think>Compare the two candidate answers.</think>\n"
        f"<answer>Overall Judgment: Answer {choice} is better.</answer>"
    )


def _with_tool_result(final_answer: str, result: str) -> str:
    return (
        "assistant\n"
        "<think>I should inspect the image.</think>\n"
        '<tool_call>{"name": "image_zoom_in_tool"}</tool_call>\n'
        "user\n"
        f"<tool_response>{result}</tool_response>\n"
        f"{final_answer}"
    )


@pytest.mark.parametrize(
    ("response", "expected_score"),
    [
        ("assistant\nAnswer 1 is better.", 0.0),
        (_final_answer(2), 0.3),
        (_with_tool_result(_final_answer(2), "Tool execution failed."), 0.4),
        (_final_answer(1), 1.0),
        (_with_tool_result(_final_answer(1), "Tool execution failed."), 1.0),
        (
            _with_tool_result(
                _final_answer(1),
                "This is the zoom-in image `observation_1`.",
            ),
            1.5,
        ),
    ],
)
def test_stage_two_reward_mapping(response: str, expected_score: float) -> None:
    result = reward_v8.compute_score_detailed(response, ground_truth=1)
    assert result["final_score"] == pytest.approx(expected_score)


def test_reward_components_match_equation_five() -> None:
    response = _with_tool_result(
        _final_answer(1),
        "This is the zoom-in image `observation_1`.",
    )

    result = reward_v8.compute_score_detailed(response, ground_truth=1)

    assert result["format_reward"] == pytest.approx(0.3)
    assert result["accuracy_reward"] == pytest.approx(0.7)
    assert result["tool_attempt_reward"] == pytest.approx(0.0)
    assert result["successful_tool_reward"] == pytest.approx(0.5)
