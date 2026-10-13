import re
import logging
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Numerical instantiation of the stage-2 reward in Eq. (5) of the paper:
#
#   R_f     = 0.3  (valid final-answer format)
#   R_try   = 0.1  (a tool was attempted when the answer is incorrect)
#   R_a     = 0.7  (the final answer is correct)
#   R_succ  = 0.5  (a successful tool result supports a correct answer)
#
# An invalid final-answer format receives zero reward.
FORMAT_REWARD = 0.3
TOOL_ATTEMPT_REWARD = 0.1
ACCURACY_REWARD = 0.7
SUCCESSFUL_TOOL_REWARD = 0.5
INVALID_FORMAT_REWARD = 0.0

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

    if not is_format_ok:
        return {
            "final_score": INVALID_FORMAT_REWARD,
            "format_reward": 0.0,
            "accuracy_reward": 0.0,
            "tool_attempt_reward": 0.0,
            "successful_tool_reward": 0.0,
            "is_correct": False,
            "is_format_ok": False,
            "predicted_answer": "",
            "expected_answer": str(ground_truth).strip().lower(),
            "reason": "Format must be <think>...</think><answer>...</answer> with no <tool_call>.",
        }

    answer_content = m.group("answer").strip()

    predicted_choice = extract_choice_from_response(answer_content)
    predicted_answer = str(predicted_choice) if predicted_choice is not None else answer_content.strip().lower()
    expected_answer = str(ground_truth).strip().lower()
    is_correct = (predicted_answer == expected_answer)

    total_tool_calls = response_str.count("<tool_response>")
    successful_tool_calls = response_str.count("This is the zoom-in image")
    has_tool_usage = total_tool_calls > 0

    # Eq. (5): R_f is awarded for a valid final format. Incorrect answers
    # receive R_try when a tool was attempted; correct answers receive R_a
    # and additionally R_succ when at least one tool call succeeded.
    format_reward = FORMAT_REWARD
    accuracy_reward = ACCURACY_REWARD if is_correct else 0.0
    tool_attempt_reward = (
        TOOL_ATTEMPT_REWARD if (not is_correct and has_tool_usage) else 0.0
    )
    successful_tool_reward = (
        SUCCESSFUL_TOOL_REWARD
        if (is_correct and successful_tool_calls > 0)
        else 0.0
    )
    final_score = (
        format_reward
        + accuracy_reward
        + tool_attempt_reward
        + successful_tool_reward
    )

    return {
        "final_score": final_score,
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "tool_attempt_reward": tool_attempt_reward,
        "successful_tool_reward": successful_tool_reward,
        "is_correct": is_correct,
        "is_format_ok": True,
        "predicted_answer": predicted_answer,
        "expected_answer": expected_answer,
        "has_tool_usage": has_tool_usage,
        "successful_tool_calls": successful_tool_calls,
        "total_tool_calls": total_tool_calls,
        "tool_bonus": tool_attempt_reward + successful_tool_reward,
    }


def compute_score(
    response_str: str, ground_truth: Union[str, int], extra_info: Dict[str, Any] = None
) -> float:
    """Return scalar reward only."""
    return compute_score_detailed(response_str, ground_truth, extra_info)["final_score"]
