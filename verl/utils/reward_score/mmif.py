import re
from typing import Optional


def extract_final_judgment(response_str: str) -> Optional[str]:
    """Extract the final True/False judgment from the response.

    Args:
        response_str: Model's complete response including tool calls and final judgment

    Returns:
        str: 'true' or 'false' if found, None otherwise
    """
    # Match <answer>Overall Judgment: True</answer> or <answer>Overall Judgment: False</answer>
    pattern = r"<answer>Overall Judgment:\s*(True|False)</answer>"
    match = re.search(pattern, response_str, re.IGNORECASE)

    if match:
        return match.group(1).lower()
    return None


def extract_tool_calls(response_str: str) -> list:
    """Extract all tool calls from the response for debugging purposes.

    Args:
        response_str: Model's complete response

    Returns:
        list: List of tool call strings found
    """
    pattern = r"<tool_call>.*?</tool_call>"
    matches = re.findall(pattern, response_str, re.DOTALL)
    return matches

# [TAG-NEW] Currently only this function is used
def compute_score(response_str: str, ground_truth: str, extra_info=None) -> float:
    """Compute score for MMIF instruction following task.

    Args:
        response_str: Model's complete response including tool calls and final judgment
        ground_truth: Expected True/False answer
        extra_info: Additional information (unused in this case)

    Returns:
        float: 1.0 if judgment matches ground_truth, 0.0 otherwise
    """
    extracted_judgment = extract_final_judgment(response_str)

    if extracted_judgment is None:
        # If no final judgment is found, return 0
        print(f"[MMIF] Warning: No final judgment found in response")
        return 0.0

    # Convert ground_truth to lowercase for comparison
    expected = str(ground_truth).lower()

    # Return 1 if correct; otherwise 0
    score = 1.0 if extracted_judgment == expected else 0.0

    print(
        f"[MMIF] Extracted judgment: {extracted_judgment}, Expected: {expected}, Score: {score}"
    )

    return score


def compute_score_with_details(
    response_str: str, ground_truth: str, extra_info=None
) -> dict:
    """Compute score with detailed information for debugging and analysis.

    Args:
        response_str: Model's complete response including tool calls and final judgment
        ground_truth: Expected True/False answer
        extra_info: Additional information (unused in this case)

    Returns:
        dict: Dictionary containing score and detailed information
    """
    extracted_judgment = extract_final_judgment(response_str)
    tool_calls = extract_tool_calls(response_str)

    if extracted_judgment is None:
        return {
            "score": 0.0,
            "extracted_judgment": None,
            "expected_judgment": str(ground_truth).lower(),
            "tool_calls_count": len(tool_calls),
            "has_final_answer": False,
            "error": "No final judgment found",
        }

    expected = str(ground_truth).lower()
    score = 1.0 if extracted_judgment == expected else 0.0

    return {
        "score": score,
        "extracted_judgment": extracted_judgment,
        "expected_judgment": expected,
        "tool_calls_count": len(tool_calls),
        "has_final_answer": True,
        "is_correct": score == 1.0,
        "error": None,
    }
