import re
import logging
from typing import Union, Dict, Any
from eval_utils.utils import (
    extract_choice_from_resp_single_rm, 
    extract_choice_from_resp_pair_rm
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ================================================================
#  Choice extraction (only allow “Answer 1/2 is better/preferred”)
# ================================================================
# def extract_choice_from_response(response_str: str) -> Union[int, None]:
#     """Extract numeric choice (1 or 2) from model response."""
#     patterns = [
#         r"Answer\s*([12])\s*is\s*(?:the\s*)?(?:slightly\s*)?(?:better|preferred|correct)",
#     ]
#     for pattern in patterns:
#         matches = list(re.finditer(pattern, response_str, re.IGNORECASE))
#         if matches:
#             try:
#                 val = int(matches[-1].group(1))
#                 if val in (1, 2):
#                     return val
#             except (ValueError, IndexError):
#                 continue
#     return None


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

    if ground_truth == -2:
        ground_truth_str = "false"
    elif ground_truth == -1:
        ground_truth_str = "true"
    else:
        ground_truth_str = str(ground_truth)

    if not is_format_ok:
        return {
            "final_score": 0.0,
            "is_correct": False,
            "is_format_ok": False,
            "predicted_answer": "",
            "expected_answer": ground_truth_str.strip().lower(),
            "reason": "Format must be <think>...</think><answer>...</answer> with no <tool_call>.",
        }

    answer_content = m.group("answer").strip()
    # print(f"answer_content: {answer_content}")

    rm_tag = extra_info.get("rm_tag", None)
    if rm_tag == "single_rm":
        predicted_choice = str(extract_choice_from_resp_single_rm(answer_content))
    elif rm_tag == "pair_rm":
        predicted_choice = str(extract_choice_from_resp_pair_rm(answer_content))
    else:
        raise ValueError(f"Invalid rm_tag: {rm_tag}")
    
    predicted_answer = str(predicted_choice).strip().lower() if predicted_choice is not None else answer_content.strip().lower()
    expected_answer = ground_truth_str.strip().lower()
    # print(f"predicted_answer: {predicted_answer}, expected_answer: {expected_answer}")
    is_correct = (predicted_answer == expected_answer)
    total_tool_calls = response_str.count("<tool_response>")
    # Note: change here
    successful_tool_calls_1 = response_str.count("Concatenated page images for query")
    successful_tool_calls_2 = response_str.count("Here is page image #")
    successful_tool_calls = successful_tool_calls_1 + successful_tool_calls_2
    has_tool_usage = total_tool_calls > 0

    if not is_correct:
        final_score = 0.4 if has_tool_usage else 0.3
        tool_bonus = 0.0
    else:
        if successful_tool_calls > 0:
            final_score = 1.5
            tool_bonus = 0.5
        else:
            final_score = 1.0
            tool_bonus = 0.0

    return {
        "final_score": final_score,
        "is_correct": is_correct,
        "is_format_ok": True,
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

if __name__ == "__main__":
    """
    Comprehensive tests for compute_score_detailed()

    We test combinations of:
      - Format OK / Bad
      - Correct / Wrong answer
      - Tool used / success / not used
      - rm_tag = single_rm / pair_rm
    """

    TEST_CASES = {
        "1: 1.0": {
            "resp": """
assistant
<think>Reasoning step here.</think>
<answer>Overall Judgment: Answer 1 is better.</answer>
""",
            "ground_truth": 1,
            "extra": {"rm_tag": "pair_rm"},
        },

        "2: 1.5": {
            "resp": """
<tool_response>Concatenated page images for query</tool_response>
assistant
<think>Analyzing query.</think>
<answer>Overall Judgment: Answer 2 is preferred.</answer>
""",
            "ground_truth": 2,
            "extra": {"rm_tag": "pair_rm"},
        },
        "3: 1.5": {
            "resp": """
<tool_response>Here is page image #1</tool_response>
assistant
<think>Steps...</think>
<answer>Overall Judgment: False.</answer>
""",
            "ground_truth": -2 ,
            "extra": {"rm_tag": "single_rm"},
        },
        "4: 0.4": {
            "resp": """
<tool_response>Here is page image #1</tool_response>
assistant
<think>Steps...</think>
<answer>Overall Judgment: False.</answer>
""",
            "ground_truth": -1 ,
            "extra": {"rm_tag": "single_rm"},
        },
    }
    print("=== New Reward Function Test Cases ===")
    for name, data in TEST_CASES.items():
        result = compute_score_detailed(
            data["resp"], data["ground_truth"], data["extra"]
        )
        print(f"{name:45s} => {result['final_score']:.2f} | "
              f"Correct={result['is_correct']} | FormatOK={result['is_format_ok']} | "
              f"Tools={result.get('total_tool_calls', 0)} | "
              f"Success={result.get('successful_tool_calls', 0)}")
