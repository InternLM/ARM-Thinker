import re
import logging
from typing import Union

# Set logger level to ensure INFO logs are visible
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_choice_from_response(response_str: str) -> Union[int, None]:
    """Extract choice number from response string.

    Args:
        response_str: The complete response from the model

    Returns:
        int: The extracted choice number (1 or 2), or None if not found
    """
    # Pattern 1: Match "Answer X is better" format
    pattern1 = r"Answer\s*(\d+)\s*is\s*(?:the\s*)?(?:slightly\s*)?(?:better|preferred)"
    matches1 = list(re.finditer(pattern1, response_str, re.IGNORECASE))
    if matches1:
        return int(matches1[-1].group(1))

    # Pattern 2: Match "<answer>X</answer>" format
    pattern2 = r"<answer>\s*(\d+)\s*</answer>"
    matches2 = list(re.finditer(pattern2, response_str, re.IGNORECASE))
    if matches2:
        return int(matches2[-1].group(1))

    # Pattern 3: Match "Overall Judgment: Answer X is better" format
    pattern3 = r"Overall\s*Judgment\s*[:：\-]?\s*Answer\s*(\d+)\s*is\s*(?:the\s*)?(?:slightly\s*)?(?:better|preferred)"
    matches3 = list(re.finditer(pattern3, response_str, re.IGNORECASE))
    if matches3:
        return int(matches3[-1].group(1))

    return None


def get_last_turn(response_str: str) -> str:
    """Get the last turn from the response string.

    Args:
        response_str: The complete response from the model

    Returns:
        str: The last turn content
    """
    import re

    parts = re.split(r"(?=user\n|assistant\n)", response_str)
    if parts:
        return parts[-1].strip()
    return ""


def is_valid_last_turn(last_turn: str) -> bool:
    """Check if the last turn has the correct format.

    Args:
        last_turn: The last turn content

    Returns:
        bool: True if format is valid, False otherwise
    """
    # Must have <think>...</think><answer>...</answer> format
    if last_turn.count("<think>") != 1 or last_turn.count("</think>") != 1:
        return False
    if last_turn.count("<answer>") != 1 or last_turn.count("</answer>") != 1:
        return False

    # Should not contain tool calls in the last turn
    if "<tool_call>" in last_turn or "</tool_call>" in last_turn:
        return False

    return True


def extract_answer_from_last_turn(last_turn: str) -> str:
    """Extract answer content from the last turn.

    Args:
        last_turn: The last turn content

    Returns:
        str: The answer content
    """
    import re

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, last_turn, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def compute_score(
    response_str: str, ground_truth: Union[str, int], extra_info=None
) -> float:
    """Compute score for deepeyes_crop dataset.

    Args:
        response_str: The complete response from the model
        ground_truth: The correct choice number (as string or int)
        extra_info: Additional information (unused in this case)

    Returns:
        float: 1.0 if the choice is correct, 0.0 otherwise
    """
    # Log the full received solution_str
    logger.info("=" * 80)
    logger.info("COMPUTE_SCORE RECEIVED SOLUTION_STR:")
    logger.info("=" * 80)
    logger.info(response_str)
    logger.info("=" * 80)
    logger.info(f"Ground truth: {ground_truth}")
    logger.info("=" * 80)

    # Get the last turn
    last_turn = get_last_turn(response_str)
    logger.info(f"Last turn: {last_turn}")

    # Check whether the last turn format is correct
    format_valid = is_valid_last_turn(last_turn)
    logger.info(f"Last turn format valid: {format_valid}")

    if not format_valid:
        logger.info("Last turn format invalid, returning 0.0")
        return 0.0

    # Extract the answer from the last turn
    answer_content = extract_answer_from_last_turn(last_turn)
    logger.info(f"Answer content: {answer_content}")

    # Extract the choice from the answer
    extracted_choice = extract_choice_from_response(answer_content)
    logger.info(f"Extracted choice: {extracted_choice}")

    if extracted_choice is None:
        logger.info("No choice extracted, returning 0.0")
        return 0.0

    # Convert ground_truth to integer for comparison
    expected_choice = int(ground_truth)
    logger.info(f"Expected choice: {expected_choice}")

    # Compare whether the choice is correct
    accuracy_score = 1.0 if extracted_choice == expected_choice else 0.0

    # Final score: format correct + answer correct
    final_score = accuracy_score * 0.7 + 0.3  # 0.3 for correct format, 0.7 for correct answer

    logger.info(f"Accuracy score: {accuracy_score}")
    logger.info(f"Final score: {final_score}")
    logger.info("=" * 80)

    return final_score
