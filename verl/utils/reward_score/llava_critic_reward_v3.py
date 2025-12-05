import re
import logging
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Helper functions (unchanged) ---
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

# --- Regex definitions (unchanged) ---
STRICT_LAST_TURN_RE = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
BANNED_TAGS_RE = re.compile(r"</?tool_call>", re.IGNORECASE)


# --- MODIFIED REWARD FUNCTION ---
def compute_score_detailed_v2(
    response_str: str, ground_truth: Union[str, int], extra_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Reward design with Reward Shaping:

      - Format OK + Correct answer:     1.5
      - Format FAIL + Correct answer:   1.0  <-- NEW
      - Format OK + Wrong answer:       0.5
      - Format FAIL + Wrong answer:     0.0
    """
    last_turn = get_last_assistant_turn(response_str)
    expected_answer_str = str(ground_truth).strip().lower()

    # 1) Check for perfect format
    m = STRICT_LAST_TURN_RE.match(last_turn)
    has_banned = BANNED_TAGS_RE.search(last_turn) is not None
    is_format_ok = bool(m and not has_banned)

    # 2) Extract choice and determine correctness
    # We attempt extraction regardless of format correctness
    # For perfect format, we extract from the <answer> tag for precision.
    # For imperfect format, we extract from the whole last turn.
    text_to_extract_from = m.group("answer").strip() if is_format_ok else last_turn
    predicted_choice = extract_choice_from_response(text_to_extract_from)
    
    is_correct = False
    if predicted_choice is not None:
        is_correct = (str(predicted_choice) == expected_answer_str)

    # 3) Assign score based on the new logic
    final_score = 0.0
    reason = ""
    # if is_format_ok:
    #     if is_correct:
    #         final_score = 1.5
    #         reason = "Format OK, Answer Correct."
    #     else:
    #         final_score = 0.5
    #         reason = "Format OK, Answer Wrong."
    # else: # Format is NOT ok
    #     if is_correct:
    #         final_score = 1.0 # Give partial credit for correct answer
    #         reason = "Format FAIL, but Answer Correct."
    #     else:
    #         final_score = 0.0
    #         reason = "Format FAIL, Answer Wrong."
    if is_correct:
        final_score = 1.0
        reason = "Answer Correct."
    else:
        final_score = 0.0
        reason = "Answer Wrong."

    return {
        "final_score": final_score,
        "is_correct": is_correct,
        "is_format_ok": is_format_ok,
        "predicted_choice": predicted_choice,
        "expected_answer": expected_answer_str,
        "reason": reason,
    }

# Wrapper to maintain original function signature
def compute_score(
    response_str: str, ground_truth: Union[str, int], extra_info: Dict[str, Any] = None
) -> float:
    """Return the final scalar reward."""
    return compute_score_detailed_v2(response_str, ground_truth, extra_info)["final_score"]


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

        # ❌ Strict format but numeric only (should be 0.5)
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

