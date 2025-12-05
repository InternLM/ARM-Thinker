import re
import logging
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ================================================================
#  Choice extraction (only allow “Answer 1/2 is better/preferred”)
# ================================================================
def extract_choice_from_response(response_str: str) -> Union[int, None]:
    """Extract numeric choice (1 or 2) from model response."""
    patterns = [
        r"Answer\s*([12])\s*is\s*(?:the\s*)?(?:slightly\s*)?(?:better|preferred|correct)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, response_str, re.IGNORECASE))
        if matches:
            try:
                val = int(matches[-1].group(1))
                if val in (1, 2):
                    return val
            except (ValueError, IndexError):
                continue
    return None


# ================================================================
#  Helpers for turn & tag extraction
# ================================================================
def get_last_assistant_turn(response_str: str) -> str:
    """Return the last assistant block."""
    parts = response_str.split("assistant\n")
    return parts[-1].strip() if len(parts) > 1 else response_str.strip()


STRICT_LAST_TURN_RE = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)

BANNED_TAGS_RE = re.compile(r"</?tool_call>", re.IGNORECASE)


def extract_answer_from_last_turn(last_turn: str) -> str:
    """Extract content inside <answer>...</answer>."""
    match = re.search(r"<answer>(.*?)</answer>", last_turn, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ================================================================
#  Main scoring logic
# ================================================================
def compute_score_detailed(
    response_str: str, ground_truth: Union[str, int], extra_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Reward design (strict; total range [0.0, 1.5]):

      - Format incorrect:                                0.0
      - Format correct + Wrong answer + No tool:         0.3
      - Format correct + Wrong answer + Tool attempted:  0.4
      - Format correct + Correct answer + No tool:       1.0
      - Format correct + Correct answer + Successful tool: 1.5
    """

    last_turn = get_last_assistant_turn(response_str)
    m = STRICT_LAST_TURN_RE.match(last_turn)
    has_strict_shape = m is not None
    has_banned = BANNED_TAGS_RE.search(last_turn) is not None
    is_format_ok = bool(has_strict_shape and not has_banned)

    # if not is_format_ok:
    #     return {
    #         "final_score": 0.0,
    #         "is_correct": False,
    #         "is_format_ok": False,
    #         "predicted_answer": "",
    #         "expected_answer": str(ground_truth).strip().lower(),
    #         "reason": "Format must be <think>...</think><answer>...</answer> with no <tool_call>.",
    #     }

    answer_content = m.group("answer").strip() if is_format_ok else last_turn

    predicted_choice = extract_choice_from_response(answer_content)
    predicted_answer = str(predicted_choice) if predicted_choice is not None else answer_content.strip().lower()
    expected_answer = str(ground_truth).strip().lower()
    is_correct = (predicted_answer == expected_answer)

    total_tool_calls = response_str.count("<tool_response>")
    successful_tool_calls = response_str.count("This is the zoom-in image")
    has_tool_usage = total_tool_calls > 0

    # Only focus on the correctness of the answer
    final_score = 1.0 if is_correct else 0.0
    tool_bonus = 0.0

    return {
        "final_score": final_score,
        "is_correct": is_correct,
        "is_format_ok": is_format_ok,
        "predicted_answer": predicted_answer,
        "expected_answer": expected_answer,
        "has_tool_usage": has_tool_usage,
        "successful_tool_calls": successful_tool_calls,
        "total_tool_calls": total_tool_calls,
        "tool_bonus": tool_bonus,
    }


def compute_score(
    response_str: str, ground_truth: Union[str, int], extra_info: Dict[str, Any] = None
) -> float:
    """Return scalar reward only."""
    return compute_score_detailed(response_str, ground_truth, extra_info)["final_score"]
