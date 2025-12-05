import logging
import os
import re
from typing import Any, Optional, List, Dict, Union
from uuid import uuid4

import nltk
from dotenv import load_dotenv
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Ensure NLTK data is available
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     try:
#         nltk.download("punkt", quiet=True)
#     except Exception as e:
#         logger.warning(f"Failed to download NLTK punkt tokenizer: {e}")
# --- Environment and NLTK Setup ---
load_dotenv()
# Ensure the NLTK data path is set. Download 'punkt' if you haven't already.
# To download:
# import nltk
# nltk.download('punkt')
if "NLTK_DATA_PATH" in os.environ:
    nltk.data.path.append(os.environ["NLTK_DATA_PATH"])
else:
    # If the environment variable is not set, you might need to specify the path manually
    # or ensure that NLTK's default download location is accessible.
    print("Warning: NLTK_DATA_PATH environment variable not set.")


# --- Helper Functions ---
def _clean_text(text: str) -> str:
    """Removes leading/trailing whitespace from each line and the whole text."""
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def normalize_text_for_matching(text: str) -> str:
    """Normalizes text for start/end matching by lowercasing and stripping punctuation."""
    text = text.strip().lower()
    text = re.sub(r"^[.…]+", "", text)  # Remove leading dots
    text = re.sub(r"[.…]+$", "", text)  # Remove trailing dots
    return text.strip()


def _match_keyword_with_hashtag_support(text: str, keyword: str) -> int:
    """Match keyword with support for hashtags and special characters."""
    # Escape special regex characters in the keyword
    escaped_keyword = re.escape(keyword)

    # If keyword starts with #, use a special pattern that handles hashtags
    if keyword.startswith("#"):
        # For hashtags, match the exact hashtag without word boundaries
        pattern = rf"{escaped_keyword}"
    else:
        # For regular keywords, use word boundaries
        pattern = rf"\b{escaped_keyword}\b"

    # Find all matches (case-insensitive)
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)


# def _resolve_from_store(instance_store: dict, response_key: str) -> str:
#     """Resolve a key like 'text_0' from the per-instance response_store.

#     Falls back to 'ground_truth' if present and key equals 'text_0'.
#     Raises ValueError if the key cannot be resolved.
#     """
#     logger.info(f"=== RESOLVE FROM STORE DEBUG ===")
#     logger.info(f"instance_store: {instance_store}")
#     logger.info(f"response_key: {response_key}")
#     logger.info(
#         f"instance_store keys: {list(instance_store.keys()) if instance_store else 'None'}"
#     )

#     response_store: dict | None = instance_store.get("response_store")
#     logger.info(f"response_store: {response_store}")
#     logger.info(f"response_store type: {type(response_store)}")

#     if response_store and isinstance(response_store, dict):
#         logger.info(
#             f"response_store keys: {list(response_store.keys()) if response_store else 'None'}"
#         )
#         if response_key in response_store:
#             result = str(response_store[response_key])
#             logger.info(f"Found {response_key}: {result[:100]}...")  # show only first 100 characters
#             logger.info(f"=== END RESOLVE FROM STORE DEBUG ===")
#             return result

#     logger.error(
#         f"Cannot resolve response key '{response_key}'. Please ensure it exists in response_store."
#     )
#     logger.info(f"=== END RESOLVE FROM STORE DEBUG ===")
#     raise ValueError(
#         f"Cannot resolve response key '{response_key}'. Please ensure it exists in response_store."
#     )
def _resolve_from_store(instance_store: dict, resp_key: str) -> str:
    """Resolve an response key like 'text_0' from the per-instance response_store."""
    texts_map: dict | None = instance_store.get("response_store", {}).get("texts_map", {})

    if texts_map and isinstance(texts_map, dict) and resp_key in texts_map:
        return str(texts_map[resp_key])

    error_msg = (
        f"Cannot resolve text response key '{resp_key}'. "
        f"Current text responses you have are: {list(texts_map.keys()) if texts_map else []}. "
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


# --- Base Instruction Following Tool Class ---
class BaseInstructionFollowingTool(BaseTool):
    """Base class for instruction following tools."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        # Create tool schema from parameters if not provided
        if tool_schema is None:
            tool_schema = self._create_tool_schema()
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def _create_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Create tool schema from the parameters class attribute."""
        if not hasattr(self, "parameters"):
            raise ValueError(
                f"Tool class {self.__class__.__name__} must have a 'parameters' attribute"
            )

        # Create function schema
        function_schema = {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        # Add parameters
        for param in self.parameters:
            param_name = param["name"]
            function_schema["parameters"]["properties"][param_name] = {
                "type": param["type"],
                "description": param.get("description", ""),
            }
            if param.get("items"):
                function_schema["parameters"]["properties"][param_name]["items"] = (
                    param["items"]
                )
            if param.get("required", False):
                function_schema["parameters"]["required"].append(param_name)

        # Create OpenAIFunctionToolSchema
        return OpenAIFunctionToolSchema(type="function", function=function_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        response_store: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance and register a response store.

        Args:
            ground_truth: Optional legacy field. If provided, will be used as fallback for 'text_0'.
            response_store: Mapping from keys like 'text_0' to actual strings to be checked.
            texts: Optional list of strings which will be auto-mapped to keys 'text_0', 'text_1', ...
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Detailed debug information
        # logger.info(f"=== CREATE METHOD DEBUG ===")
        # logger.info(f"Tool class: {self.__class__.__name__}")
        # logger.info(f"instance_id: {instance_id}")
        # logger.info(f"response_store: {response_store}")
        # logger.info(f"response_store type: {type(response_store)}")
        # logger.info(f"kwargs: {kwargs}")
        # logger.info(f"kwargs keys: {list(kwargs.keys()) if kwargs else 'None'}")

        # Check whether response_store exists in kwargs
        # if "response_store" in kwargs:
        #     logger.info(f"response_store found in kwargs: {kwargs['response_store']}")

        # Print call stack information
        # import traceback

        # logger.info(f"Call stack:\n{traceback.format_stack()}")
        # logger.info(f"=== END CREATE METHOD DEBUG ===")

        self._instance_dict[instance_id] = {
            "response_store": kwargs.get("create_kwargs", {}).get("response_store", {}),
        }
        logger.info(f"self._instance_dict[instance_id][response_store]: {self._instance_dict[instance_id]['response_store']}")

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        # Execute the tool logic and get result
        result = await self._execute_logic(instance_id, parameters)

        # Return the result as text response to the model
        response_text = f"Check result: {result}"
        return ToolResponse(text=response_text), 0.0, {}

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        """Execute the tool-specific logic. Override in subclasses."""
        logger.info(f"=== EXECUTE LOGIC DEBUG ===")
        logger.info(f"Tool: {self.__class__.__name__}")
        logger.info(f"instance_id: {instance_id}")
        logger.info(f"parameters: {parameters}")
        logger.info(f"Available instances: {list(self._instance_dict.keys())}")
        if instance_id in self._instance_dict:
            logger.info(f"Instance data: {self._instance_dict[instance_id]}")
        else:
            logger.error(f"Instance {instance_id} not found in _instance_dict!")
        logger.info(f"=== END EXECUTE LOGIC DEBUG ===")

        return False

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward - not used for instruction following tools."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


# --- Tool Implementations ---


class ParagraphNumberInRangeTool(BaseInstructionFollowingTool):
    """Checks if the total number of paragraphs in the response is within a specified range."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "lower_bound",
            "type": "integer",
            "description": "The minimum allowed number of paragraphs.",
            "required": True,
        },
        {
            "name": "upper_bound",
            "type": "integer",
            "description": "The maximum allowed number of paragraphs.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        paragraphs = [p for p in re.split(r"\n\s*\n", response_clean) if p.strip()]
        actual_count = len(paragraphs)
        result = parameters["lower_bound"] <= actual_count <= parameters["upper_bound"]
        return result


class SentenceNumberInRangeTool(BaseInstructionFollowingTool):
    """Checks if the total number of sentences in the response is within a specified range."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "lower_bound",
            "type": "integer",
            "description": "The minimum allowed number of sentences.",
            "required": True,
        },
        {
            "name": "upper_bound",
            "type": "integer",
            "description": "The maximum allowed number of sentences.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        sentences = nltk.sent_tokenize(response_clean)
        actual_count = len(sentences)
        result = parameters["lower_bound"] <= actual_count <= parameters["upper_bound"]
        return result


class EachParagraphSentenceNumberInRangeTool(BaseInstructionFollowingTool):
    """Checks if the sentence count of each paragraph is within a specified range.
    Automatically detects poetry vs prose and checks lines for poetry, sentences for prose.
    """

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "lower_bound",
            "type": "integer",
            "description": "The minimum allowed sentences per paragraph (or lines for poetry).",
            "required": True,
        },
        {
            "name": "upper_bound",
            "type": "integer",
            "description": "The maximum allowed sentences per paragraph (or lines for poetry).",
            "required": True,
        },
    ]

    def _is_poetry(self, text: str) -> bool:
        """Detect if text appears to be poetry based on formatting patterns."""
        lines = text.split("\n")
        if len(lines) < 2:
            return False

        # Check poetry characteristics
        short_lines = sum(
            1 for line in lines if len(line.strip()) < 60
        )  # short line (poetry characteristic)
        dash_ends = sum(
            1 for line in lines if line.strip().endswith("-")
        )  # ends with hyphen (poetry characteristic)
        repeated_patterns = sum(
            1 for line in lines if line.strip().endswith(" -")
        )  # repeated hyphen pattern

        # Compute ratio of poetry characteristics
        total_lines = len(lines)
        poetry_score = (
            short_lines / total_lines * 0.4
            + dash_ends / total_lines * 0.4
            + repeated_patterns / total_lines * 0.2
        )

        # If poetry characteristics exceed 60%, consider it poetry
        return poetry_score > 0.6

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        lower_bound = parameters["lower_bound"]
        upper_bound = parameters["upper_bound"]

        paragraphs = [p for p in re.split(r"\n\s*\n", response_clean) if p.strip()]

        for paragraph in paragraphs:
            if self._is_poetry(paragraph):
                # Poetry mode: check number of lines
                lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
                if not (lower_bound <= len(lines) <= upper_bound):
                    return False
            else:
                # Prose mode: check number of sentences
                sentences = nltk.sent_tokenize(paragraph)
                if not (lower_bound <= len(sentences) <= upper_bound):
                    return False
        return True


class EachParagraphSentenceNumberInRangeListTool(BaseInstructionFollowingTool):
    """Checks if each paragraph's sentence count matches a corresponding range in a list."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "ranges",
            "type": "array",
            "description": "A list of [lower_bound, upper_bound] pairs, one for each paragraph.",
            "items": {"type": "array", "items": {"type": "integer"}},
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        ranges = parameters["ranges"]

        paragraphs = [p for p in re.split(r"\n\s*\n", response_clean) if p.strip()]
        if len(paragraphs) != len(ranges):
            return False

        for paragraph, (lower_bound, upper_bound) in zip(paragraphs, ranges):
            sentences = nltk.sent_tokenize(paragraph)
            if not (lower_bound <= len(sentences) <= upper_bound):
                return False
        return True


class WordCountInRangeTool(BaseInstructionFollowingTool):
    """Checks if the total word count of the response is within a specified range."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "lower_bound",
            "type": "integer",
            "description": "The minimum allowed word count.",
            "required": True,
        },
        {
            "name": "upper_bound",
            "type": "integer",
            "description": "The maximum allowed word count.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        words = response_clean.split()
        actual_count = len(words)
        # print(f"actual_count: {actual_count}")
        result = parameters["lower_bound"] <= actual_count <= parameters["upper_bound"]
        return result


class EachParagraphWordCountInRangeTool(BaseInstructionFollowingTool):
    """Checks if the word count of each paragraph is within a specified range."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "lower_bound",
            "type": "integer",
            "description": "The minimum allowed word count per paragraph.",
            "required": True,
        },
        {
            "name": "upper_bound",
            "type": "integer",
            "description": "The maximum allowed word count per paragraph.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        lower_bound = parameters["lower_bound"]
        upper_bound = parameters["upper_bound"]

        paragraphs = [p for p in re.split(r"\n\s*\n", response_clean) if p.strip()]
        for paragraph in paragraphs:
            words = paragraph.split()
            if not (lower_bound <= len(words) <= upper_bound):
                return False
        return True


class NotContainSubstringsTool(BaseInstructionFollowingTool):
    """Checks if the response does not contain any of the substrings from a given list."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "substrings",
            "type": "array",
            "description": "A list of substrings that should not appear in the response.",
            "items": {"type": "string"},
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        substrings = parameters["substrings"]
        for substring in substrings:
            if substring.lower() in response_clean.lower():
                return False
        return True


class NotContainSubstringTool(BaseInstructionFollowingTool):
    """Checks if the response does not contain a specific substring."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "substring",
            "type": "string",
            "description": "The substring that should not appear in the response.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        substring = parameters["substring"]
        result = substring.lower() not in response_clean.lower()
        return result


class EachSentenceBeginsWithTool(BaseInstructionFollowingTool):
    """Checks if every sentence begins with a specific substring (case-insensitive)."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "substring",
            "type": "string",
            "description": "The substring that each sentence must start with.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        substring_norm = normalize_text_for_matching(parameters["substring"])
        sentences = nltk.sent_tokenize(response)

        for sentence in sentences:
            sentence_norm = normalize_text_for_matching(sentence)
            if not sentence_norm.startswith(substring_norm):
                return False
        return True


class EachParagraphBeginsWithTool(BaseInstructionFollowingTool):
    """Checks if every paragraph begins with a specific substring (case-insensitive)."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "substring",
            "type": "string",
            "description": "The substring that each paragraph must start with.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        substring_norm = normalize_text_for_matching(parameters["substring"])
        paragraphs = [p for p in re.split(r"\n\s*\n", response) if p.strip()]

        for paragraph in paragraphs:
            paragraph_norm = normalize_text_for_matching(paragraph)
            if not paragraph_norm.startswith(substring_norm):
                return False
        return True


class EachParagraphEndsWithTool(BaseInstructionFollowingTool):
    """Checks if every paragraph ends with a specific substring (case-insensitive)."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "substring",
            "type": "string",
            "description": "The substring that each paragraph must end with.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        substring_norm = normalize_text_for_matching(parameters["substring"])
        paragraphs = [p for p in re.split(r"\n\s*\n", response) if p.strip()]

        for paragraph in paragraphs:
            paragraph_norm = normalize_text_for_matching(paragraph)
            if not paragraph_norm.endswith(substring_norm):
                return False
        return True


class EachSentenceEndsWithTool(BaseInstructionFollowingTool):
    """Checks if every sentence ends with a specific substring (case-insensitive)."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "substring",
            "type": "string",
            "description": "The substring that each sentence must end with.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        substring_norm = normalize_text_for_matching(parameters["substring"])
        sentences = nltk.sent_tokenize(response)

        for sentence in sentences:
            sentence_norm = normalize_text_for_matching(sentence)
            if not sentence_norm.endswith(substring_norm):
                return False
        return True


class ResponseBeginsWithTool(BaseInstructionFollowingTool):
    """Checks if the entire response begins with a specific substring (case-insensitive)."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "substring",
            "type": "string",
            "description": "The substring that the response must start with.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_norm = normalize_text_for_matching(response)
        substring_norm = normalize_text_for_matching(parameters["substring"])
        result = response_norm.startswith(substring_norm)
        return result


class ResponseEndsWithTool(BaseInstructionFollowingTool):
    """Checks if the entire response ends with a specific substring (case-insensitive)."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "substring",
            "type": "string",
            "description": "The substring that the response must end with.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_norm = normalize_text_for_matching(response)
        substring_norm = normalize_text_for_matching(parameters["substring"])
        result = response_norm.endswith(substring_norm)
        return result


class EachKeywordMentionedInRangeTool(BaseInstructionFollowingTool):
    """Checks if the count of each individual keyword from a list falls within a specified range."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "keywords",
            "type": "array",
            "description": "A list of keywords to count.",
            "items": {"type": "string"},
            "required": True,
        },
        {
            "name": "lower_bound_times",
            "type": "integer",
            "description": "The minimum number of times each keyword must appear.",
            "required": True,
        },
        {
            "name": "upper_bound_times",
            "type": "integer",
            "description": "The maximum number of times each keyword must appear.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        keywords = parameters["keywords"]
        lower_bound = parameters["lower_bound_times"]
        upper_bound = parameters["upper_bound_times"]

        for keyword in keywords:
            count = _match_keyword_with_hashtag_support(response_clean, keyword)
            if not (lower_bound <= count <= upper_bound):
                return False
        return True


class TotalKeywordsMentionedInRangeTool(BaseInstructionFollowingTool):
    """Checks if the total count of all keywords from a list falls within a specified range."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "keywords",
            "type": "array",
            "description": "A list of keywords to count.",
            "items": {"type": "string"},
            "required": True,
        },
        {
            "name": "lower_bound_times",
            "type": "integer",
            "description": "The minimum total number of times the keywords must appear.",
            "required": True,
        },
        {
            "name": "upper_bound_times",
            "type": "integer",
            "description": "The maximum total number of times the keywords must appear.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        keywords = parameters["keywords"]
        lower_bound = parameters["lower_bound_times"]
        upper_bound = parameters["upper_bound_times"]

        total_count = 0
        for keyword in keywords:
            count = _match_keyword_with_hashtag_support(response_clean, keyword)
            total_count += count

        result = lower_bound <= total_count <= upper_bound
        return result


class PercentagePrecisionTool(BaseInstructionFollowingTool):
    """Checks if all numbers followed by a '%' sign have a specific number of decimal places."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "precision",
            "type": "integer",
            "description": "The required number of decimal places for percentages.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        precision = parameters["precision"]

        # Find all percentage numbers
        pattern = r"(\d+\.\d+)%"
        matches = re.findall(pattern, response_clean)

        for match in matches:
            decimal_places = len(match.split(".")[1])
            if decimal_places != precision:
                return False
        return True


class NumberPrecisionTool(BaseInstructionFollowingTool):
    """Checks if all numeric values have a specific number of decimal places."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
        {
            "name": "precision",
            "type": "integer",
            "description": "The required number of decimal places for all numbers.",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)
        precision = parameters["precision"]

        # Find all decimal numbers
        pattern = r"(\d+\.\d+)"
        matches = re.findall(pattern, response_clean)

        for match in matches:
            decimal_places = len(match.split(".")[1])
            if decimal_places != precision:
                return False
        return True


class NoArabicNumberTool(BaseInstructionFollowingTool):
    """Checks if the response contains no standalone Arabic numbers."""

    parameters = [
        {
            "name": "response",
            "type": "string",
            "description": "Text name to be checked (e.g., 'text_0', 'text_1', 'text_2', etc.)",
            "required": True,
        },
    ]

    async def _execute_logic(
        self, instance_id: str, parameters: dict[str, Any]
    ) -> bool:
        response = _resolve_from_store(
            self._instance_dict[instance_id], parameters["response"]
        )
        response_clean = _clean_text(response)

        # Check for standalone Arabic numbers (not part of words)
        pattern = r"\b\d+\b"
        matches = re.findall(pattern, response_clean)

        return len(matches) == 0
