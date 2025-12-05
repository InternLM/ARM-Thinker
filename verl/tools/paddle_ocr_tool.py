import asyncio
import logging
import os
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, List, Dict, Union
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Set up PaddleOCR environment
os.environ["HF_HUB_OFFLINE"] = "true"
os.environ["PADDLEX_HOME"] = "/path/to/your/.paddlex"

# Local model directory
base = pathlib.Path(os.environ["PADDLEX_HOME"]) / "official_models"

try:
    from paddleocr import PaddleOCR
except ImportError:
    logger.error(
        "PaddleOCR is not installed. Please install it with: pip install paddleocr"
    )
    PaddleOCR = None
except Exception as e:
    logger.warning(f"Unexpected error importing PaddleOCR: {e}")
    PaddleOCR = None


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


# Global OCR instance and concurrency control
_global_ocr_instance = None
_ocr_lock = threading.Lock()
_ocr_semaphore = None  # Will be initialized with max concurrent requests
_thread_pool = None  # Will be initialized for OCR operations

# Configuration for concurrency control
MAX_CONCURRENT_OCR_REQUESTS = 16


def _initialize_ocr_resources():
    """Initialize OCR resources with concurrency control."""
    global _ocr_semaphore, _thread_pool

    if _ocr_semaphore is None:
        _ocr_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR_REQUESTS)

    if _thread_pool is None:
        # Use thread pool with max concurrent requests
        _thread_pool = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_OCR_REQUESTS, thread_name_prefix="ocr_worker"
        )


def _get_ocr_instance():
    """Get or create the global OCR instance with thread safety."""
    global _global_ocr_instance

    if _global_ocr_instance is None and PaddleOCR is not None:
        with _ocr_lock:
            # Double-check locking pattern
            if _global_ocr_instance is None:
                try:
                    logger.info(f"Creating OCR instance in process {os.getpid()}")
                    _global_ocr_instance = PaddleOCR(
                        doc_orientation_classify_model_dir=str(
                            base / "PP-LCNet_x1_0_doc_ori"
                        ),
                        textline_orientation_model_dir=str(
                            base / "PP-LCNet_x1_0_textline_ori"
                        ),
                        text_detection_model_dir=str(base / "PP-OCRv5_server_det"),
                        text_recognition_model_dir=str(base / "PP-OCRv5_server_rec"),
                        use_doc_orientation_classify=True,
                        use_doc_unwarping=False,
                        use_textline_orientation=True,
                    )
                    logger.info(
                        f"OCR instance created successfully in process {os.getpid()}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create OCR instance: {e}")
                    _global_ocr_instance = None
    return _global_ocr_instance


async def _run_ocr_inference(image_path: str) -> str:
    """Run OCR inference in a separate thread with error handling."""
    if _thread_pool is None:
        _initialize_ocr_resources()

    ocr = _get_ocr_instance()
    if ocr is None:
        return "Error: PaddleOCR is not available. Please install paddleocr."

    def _ocr_worker():
        """Worker function to run OCR in thread pool."""
        try:
            # Run OCR inference
            result = ocr.predict(input=image_path)

            # Extract text from results
            extracted_texts = []
            for res in result:
                texts = res.get("rec_texts", [])
                extracted_texts.extend(texts)

            # Join all texts into a single paragraph
            paragraph = " ".join(extracted_texts)
            if not paragraph.strip():
                return "No text content detected in the image."
            return paragraph

        except Exception as e:
            error_msg = f"Error processing image {image_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    try:
        # Run OCR in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_thread_pool, _ocr_worker)
        return result
    except Exception as e:
        error_msg = f"Unexpected error during OCR: {str(e)}"
        logger.error(error_msg)
        return error_msg


class BasePaddleOCRTool(BaseTool):
    """Base class for PaddleOCR tools."""

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
        response_text = f"OCR result: {result}"
        return ToolResponse(text=response_text), 0.0, {}

    async def _execute_logic(self, instance_id: str, parameters: dict[str, Any]) -> str:
        """Execute the tool-specific logic. Override in subclasses."""
        return ""

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward - not used for OCR tools."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class ExtractTextFromImageTool(BasePaddleOCRTool):
    """Extracts text from an image using PaddleOCR."""

    parameters = [
        {
            "name": "image",
            "type": "string",
            "description": "Image name to be processed (e.g., 'image_0', 'image_1', 'image_2', etc.)",
            "required": True,
        },
    ]

    async def _execute_logic(self, instance_id: str, parameters: dict[str, Any]) -> str:
        # Initialize OCR resources if needed
        if _ocr_semaphore is None:
            _initialize_ocr_resources()

        image_key = parameters["image"]
        image_path = _resolve_from_store(self._instance_dict[instance_id], image_key)

        logger.info(f"OCR tool called for image: {image_key}")

        # Use semaphore to limit concurrent OCR requests
        async with _ocr_semaphore:
            try:
                result = await _run_ocr_inference(image_path)
                logger.info(f"OCR completed for {image_key}")
                return result
            except Exception as e:
                error_msg = f"Error processing image {image_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg
