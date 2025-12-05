import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, List, Dict, Union
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Set up Pix2Text environment
os.environ["HF_HUB_OFFLINE"] = "0"

try:
    from pix2text import Pix2Text
except ImportError:
    logger.error(
        "Pix2Text is not installed. Please install it with: pip install pix2text"
    )
    Pix2Text = None


def _resolve_from_store(instance_store: dict, image_key: str) -> str:
    """Resolve an image key like 'image_0' from the per-instance response_store.

    Raises ValueError if the key cannot be resolved.
    """
    response_store: dict | None = instance_store.get("response_store")

    if response_store and isinstance(response_store, dict):
        if image_key in response_store:
            return str(response_store[image_key])

    logger.error(
        f"Cannot resolve image key '{image_key}'. Please ensure it exists in response_store."
    )
    raise ValueError(
        f"Cannot resolve image key '{image_key}'. Please ensure it exists in response_store."
    )


# Global Pix2Text instance and concurrency control
_global_p2t_instance = None
_p2t_lock = threading.Lock()
_p2t_semaphore = None  # Will be initialized with max concurrent requests
_thread_pool = None  # Will be initialized for Pix2Text operations

# Configuration for concurrency control
MAX_CONCURRENT_P2T_REQUESTS = 16


def _initialize_p2t_resources():
    """Initialize Pix2Text resources with concurrency control."""
    global _p2t_semaphore, _thread_pool

    if _p2t_semaphore is None:
        _p2t_semaphore = asyncio.Semaphore(MAX_CONCURRENT_P2T_REQUESTS)

    if _thread_pool is None:
        # Use thread pool with max concurrent requests
        _thread_pool = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_P2T_REQUESTS, thread_name_prefix="p2t_worker"
        )


def _get_p2t_instance():
    """Get or create the global Pix2Text instance with thread safety."""
    global _global_p2t_instance

    if _global_p2t_instance is None and Pix2Text is not None:
        with _p2t_lock:
            # Double-check locking pattern
            if _global_p2t_instance is None:
                try:
                    logger.info(f"Creating Pix2Text instance in process {os.getpid()}")
                    # Use CPU by default for better compatibility
                    _global_p2t_instance = Pix2Text(device="cpu")
                    logger.info(
                        f"Pix2Text instance created successfully in process {os.getpid()}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create Pix2Text instance: {e}")
                    _global_p2t_instance = None
    return _global_p2t_instance


async def _run_p2t_inference(image_path: str, file_type: str = "text_formula") -> str:
    """Run Pix2Text inference in a separate thread with error handling."""
    if _thread_pool is None:
        _initialize_p2t_resources()

    p2t = _get_p2t_instance()
    if p2t is None:
        return "Error: Pix2Text is not available. Please install pix2text."

    def _p2t_worker():
        """Worker function to run Pix2Text in thread pool."""
        try:
            # Run Pix2Text inference
            result = p2t(image_path, file_type=file_type)

            # Extract all text content from results
            extracted_texts = []
            if hasattr(result, "elements"):
                for element in result.elements:
                    if hasattr(element, "text") and element.text:
                        extracted_texts.append(element.text)

            # Join all texts into a single result
            if extracted_texts:
                result_text = "\n".join(extracted_texts)
                return result_text
            else:
                return "No text content detected in the image."

        except Exception as e:
            error_msg = f"Error processing image {image_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    try:
        # Run Pix2Text in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_thread_pool, _p2t_worker)
        return result
    except Exception as e:
        error_msg = f"Unexpected error during Pix2Text: {str(e)}"
        logger.error(error_msg)
        return error_msg


class BasePix2TextTool(BaseTool):
    """Base class for Pix2Text tools."""

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
            response_store: Mapping from keys like 'image_0' to actual image paths to be processed.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Get response store from kwargs
        new_response_store = kwargs.get("create_kwargs", {}).get(
            "response_store", response_store or {}
        )

        self._instance_dict[instance_id] = {
            "response_store": new_response_store,
        }

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        # Execute the tool logic and get result
        result = await self._execute_logic(instance_id, parameters)

        # Return the result as text response to the model
        response_text = f"Mathematical formula extraction result: {result}"
        return ToolResponse(text=response_text), 0.0, {}

    async def _execute_logic(self, instance_id: str, parameters: dict[str, Any]) -> str:
        """Execute the tool-specific logic. Override in subclasses."""
        return ""

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward - not used for formula extraction tools."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class ExtractMathematicalFormulaTool(BasePix2TextTool):
    """Extracts mathematical formulas from images using Pix2Text."""

    parameters = [
        {
            "name": "image",
            "type": "string",
            "description": "Image name to be processed (e.g., 'image_0', 'image_1', 'image_2', etc.)",
            "required": True,
        },
    ]

    async def _execute_logic(self, instance_id: str, parameters: dict[str, Any]) -> str:
        # Initialize Pix2Text resources if needed
        if _p2t_semaphore is None:
            _initialize_p2t_resources()

        image_key = parameters["image"]
        image_path = _resolve_from_store(self._instance_dict[instance_id], image_key)

        logger.info(f"Pix2Text tool called for image: {image_key}")

        # Use semaphore to limit concurrent Pix2Text requests
        async with _p2t_semaphore:
            try:
                result = await _run_p2t_inference(image_path, file_type="text_formula")
                logger.info(f"Pix2Text completed for {image_key}")
                return result
            except Exception as e:
                error_msg = f"Error processing image {image_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg


class ExtractFormulaOnlyTool(BasePix2TextTool):
    """Extracts only mathematical formulas from images using Pix2Text (formula-only mode)."""

    parameters = [
        {
            "name": "image",
            "type": "string",
            "description": "Image name to be processed (e.g., 'image_0', 'image_1', 'image_2', etc.)",
            "required": True,
        },
    ]

    async def _execute_logic(self, instance_id: str, parameters: dict[str, Any]) -> str:
        # Initialize Pix2Text resources if needed
        if _p2t_semaphore is None:
            _initialize_p2t_resources()

        image_key = parameters["image"]
        image_path = _resolve_from_store(self._instance_dict[instance_id], image_key)

        logger.info(f"Pix2Text formula tool called for image: {image_key}")

        # Use semaphore to limit concurrent Pix2Text requests
        async with _p2t_semaphore:
            try:
                result = await _run_p2t_inference(image_path, file_type="formula")
                logger.info(f"Pix2Text formula completed for {image_key}")
                return result
            except Exception as e:
                error_msg = f"Error processing image {image_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg
