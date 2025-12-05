import re
import logging
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Choice extraction: only allow 1 or 2 ---
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

def get_last_assistant_turn(response_str: str) -> str:
    """Return the last assistant block."""
    parts = response_str.split("assistant\n")
    return parts[-1].strip() if len(parts) > 1 else response_str.strip()

# --- Strict format check: ^\s*<think>...</think>\s*<answer>...</answer>\s*$ ---
STRICT_LAST_TURN_RE = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)

BANNED_TAGS_RE = re.compile(r"</?tool_call>", re.IGNORECASE)

def compute_score_detailed(
    response_str: str, ground_truth: Union[str, int], extra_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Reward design (tool-independent, same range as original [0.0, 1.5]):

      - Format incorrect:                0.0
      - Format correct + Wrong answer:   0.3
      - Format correct + Correct answer: 1.5
    """

    # 1) Strict format validation on the *last assistant turn* only
    last_turn = get_last_assistant_turn(response_str)

    m = STRICT_LAST_TURN_RE.match(last_turn)
    has_strict_shape = m is not None
    has_banned = BANNED_TAGS_RE.search(last_turn) is not None
    is_format_ok = bool(has_strict_shape and not has_banned)

    if not is_format_ok:
        return {
            "final_score": 0.0,
            "is_correct": False,
            "is_format_ok": False,
            "predicted_answer": "",
            "expected_answer": str(ground_truth).strip().lower(),
            "reason": "Format must be <think>...</think><answer>...</answer> with no tool_* tags.",
        }

    # 2) Extract answer text (from the strict match's <answer>...</answer>)
    answer_content = m.group("answer").strip()
    print(f"answer_content: {answer_content}")

    # 3) Correctness: only accept choice 1/2
    predicted_choice = extract_choice_from_response(answer_content)
    if predicted_choice is None:
        is_correct = False
    else:
        predicted_answer = str(predicted_choice)
        is_correct = (predicted_answer == str(ground_truth).strip().lower())

    # 4) Reward
    final_score = 1.5 if is_correct else 0.3

    return {
        "final_score": final_score,
        "is_correct": is_correct,
        "is_format_ok": True,
        "predicted_choice": predicted_choice,
        "expected_answer": str(ground_truth).strip().lower(),
    }

def compute_score(
    response_str: str, ground_truth: Union[str, int], extra_info: Dict[str, Any] = None
) -> float:
    """Return the final scalar reward."""
    return compute_score_detailed(response_str, ground_truth, extra_info)["final_score"]


if __name__ == "__main__":
    cases = {
        # ✅ Fully correct: strict format + correct answer
        "✅ Correct + Strict Format (Canonical)": """
assistant
<think>Reason carefully.</think>
<answer>Overall Judgment: Answer 1 is better.</answer>
""",

        # ✅ Strict format + correct answer (alternative 'preferred' wording)
        "✅ Correct + Strict Format (Preferred wording)": """
assistant
<think>Check both responses and decide.</think>
<answer>Overall Judgment: Answer 1 is preferred.</answer>
""",

        # ❌ Strict format + wrong answer (wrong number)
        "❌ Wrong + Strict Format": """
assistant
<think>Reason incorrectly.</think>
<answer>Overall Judgment: Answer 2 is preferred.</answer>
""",

        # ❌ Strict format but incomplete sentence (missing predicate)
        "❌ Wrong + Strict Format (Truncated judgment)": """
assistant
<think>Reason carefully.</think>
<answer>Overall Judgment: Answer 1</answer>
""",

        # ❌ Strict format but numeric only (should be 0.3)
        "❌ Wrong + Strict Format (Numeric only)": """
assistant
<think>Reason carefully.</think>
<answer>1</answer>
""",

        # ⚠️ Strict format but semantically invalid (missing 'Answer' keyword)
        "⚠️ Format OK + Invalid wording": """
assistant
<think>Compare reasoning chains.</think>
<answer>The first option seems better overall.</answer>
""",

        # ⛔ Missing <think> (violates strict structure)
        "⛔ Not strict (missing think)": """
assistant
<answer>Overall Judgment: Answer 1 is better.</answer>
""",

        # ⛔ Missing <answer> (violates strict structure)
        "⛔ Not strict (missing answer)": """
assistant
<think>Reasoning here.</think>
""",

        # ⛔ Wrong tag order (<answer> before <think>)
        "⛔ Wrong tag order": """
assistant
<answer>Overall Judgment: Answer 1 is better.</answer>
<think>Reason carefully.</think>
""",

        # ⛔ Extra text (before <think>)
        "⛔ Bad Format (extra text before think)": """
assistant
Intro text
<think>Reason carefully.</think>
<answer>Overall Judgment: Answer 1 is better.</answer>
""",

        # ⛔ Contains tool_call tag
        "⛔ Contains tool_call": """
assistant
<think>Reason carefully.
<tool_call>{"name":"image_zoom_in_tool"}</tool_call></think>
<answer>Overall Judgment: Answer 1 is better.</answer>
""",

        # ⛔ Contains tool_response tag
        "⛔ Contains tool_response": """
assistant
<think>Reason carefully.</think>
<answer>Overall Judgment: Answer 1 is better.</answer>
<tool_response>This is the zoom-in image.</tool_response>
""",

        # ⛔ Extra unexpected tags (e.g., system directive)
        "⛔ Extra unexpected tags": """
assistant
<think>Reason carefully.</think>
<answer>Overall Judgment: Answer 1 is better.</answer>
<extra_tag>debug info</extra_tag>
""",

        # ⚠️ Empty answer block (strict format but invalid content)
        "⚠️ Empty answer block": """
assistant
<think>Reason carefully.</think>
<answer>   </answer>
""",
    }

    print("=== Reward Function Test Cases ===")
    for name, text in cases.items():
        score = compute_score(text, 1)
        print(f"{name:45s} => {score:.2f}")

