import asyncio
import logging
import os
import pathlib
import threading
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, List, Dict, Union
from uuid import uuid4
from pathlib import Path

from verl.utils.rollout_trace import rollout_trace_op
from dotenv import load_dotenv
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# load from .env
load_dotenv()
MINERU_SERVER_URL = os.getenv("MINERU_SERVER_URL", "http://10.102.216.44:30000")

# Set up environment for mineru
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    from mineru.cli.common import (
        convert_pdf_bytes_to_bytes_by_pypdfium2,
        prepare_env,
        read_fn,
    )
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
    from mineru.backend.vlm.vlm_middle_json_mkcontent import (
        union_make as vlm_union_make,
    )
    from mineru.utils.enum_class import MakeMode
    from mineru.utils.draw_bbox import draw_layout_bbox
except ImportError:
    logger.error("MinerU is not installed. Please install it with: pip install mineru")
    # Set dummy functions to avoid errors
    convert_pdf_bytes_to_bytes_by_pypdfium2 = None
    prepare_env = None
    read_fn = None
    FileBasedDataWriter = None
    vlm_doc_analyze = None
    vlm_union_make = None
    MakeMode = None
    draw_layout_bbox = None
except Exception as e:
    logger.warning(f"Unexpected error importing MinerU: {e}")
    # Set dummy functions to avoid errors
    convert_pdf_bytes_to_bytes_by_pypdfium2 = None
    prepare_env = None
    read_fn = None
    FileBasedDataWriter = None
    vlm_doc_analyze = None
    vlm_union_make = None
    MakeMode = None
    draw_layout_bbox = None


def _resolve_from_store(instance_store: dict, image_key: str) -> str:
    """Resolve an image key like 'original_image' from the per-instance response_store.

    Raises ValueError if the key cannot be resolved.
    """
    imgs_map: dict | None = instance_store.get("response_store", {}).get("imgs_map", {})

    if imgs_map and isinstance(imgs_map, dict):
        if image_key in imgs_map:
            return str(imgs_map[image_key])

    error_msg = (
        f"Cannot resolve image key '{image_key}'. "
        f"Current images you have are: {list(imgs_map.keys()) if imgs_map else []}. "
        "You can call this tool like this: "
        '<tool_call>\n{{"name": "extract_text_from_image_with_mineru", '
        '"arguments": {{"image": "original_image"}}}}\n</tool_call>'
    )

    logger.error(
        error_msg
    )
    raise ValueError(
        error_msg
    )


# Global concurrency control
_vlm_semaphore = None
_thread_pool = None

# Configuration for concurrency control
MAX_CONCURRENT_VLM_REQUESTS = 16  # VLM requests are more resource intensive

# Return mode configuration - can be changed here
# Options: "md" | "content_list_json" | "middle_json"
# - "md": Returns markdown string (default)
# - "content_list_json": Returns structured content list as JSON string
# - "middle_json": Returns raw middle JSON from VLM analysis
DEFAULT_RETURN_MODE = "md"


def _initialize_vlm_resources():
    """Initialize VLM resources with concurrency control."""
    global _vlm_semaphore, _thread_pool

    if _vlm_semaphore is None:
        _vlm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VLM_REQUESTS)

    if _thread_pool is None:
        # Use thread pool with max concurrent requests
        _thread_pool = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_VLM_REQUESTS, thread_name_prefix="vlm_worker"
        )


async def _run_vlm_inference(
    image_path: str,
    server_url: str = "http://127.0.0.1:30000",
    return_type: str = DEFAULT_RETURN_MODE,
) -> str:
    """Run VLM inference in a separate thread with error handling."""
    if _thread_pool is None:
        _initialize_vlm_resources()

    if vlm_doc_analyze is None:
        return "Error: MinerU is not available. Please install mineru."

    def _vlm_worker():
        """Worker function to run VLM in thread pool."""
        try:
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(image_path)
                base = path.stem

                file_bytes = read_fn(path)
                if path.suffix.lower() == ".pdf":
                    file_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                        file_bytes, 0, None
                    )

                local_image_dir, local_md_dir = prepare_env(temp_dir, base, "auto")
                image_writer = FileBasedDataWriter(local_image_dir)
                # Note: We don't create md_writer since we don't want to save files

                # Call remote VLM-sglang-client
                middle_json, infer_result = vlm_doc_analyze(
                    file_bytes,
                    image_writer=image_writer,
                    backend="sglang-client",
                    server_url=server_url,
                )

                pdf_info = middle_json["pdf_info"]

                # Build outputs based on return_type
                if return_type == "md":
                    result = vlm_union_make(
                        pdf_info, MakeMode.MM_MD, os.path.basename(local_image_dir)
                    )
                    if not result.strip():
                        return "No text content detected in the image."
                    return result
                elif return_type == "content_list_json":
                    result = vlm_union_make(
                        pdf_info,
                        MakeMode.CONTENT_LIST,
                        os.path.basename(local_image_dir),
                    )
                    return json.dumps(result, ensure_ascii=False, indent=4)
                elif return_type == "middle_json":
                    return json.dumps(middle_json, ensure_ascii=False, indent=2)
                else:
                    raise ValueError(f"Invalid return_type: {return_type}")

        except Exception as e:
            error_msg = f"Error processing image {image_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    try:
        # Run VLM in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_thread_pool, _vlm_worker)
        return result
    except Exception as e:
        error_msg = f"Unexpected error during VLM processing: {str(e)}"
        logger.error(error_msg)
        return error_msg


class BaseMineruVLMOCRTool(BaseTool):
    """Base class for MinerU VLM OCR tools."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

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
            response_store: Mapping from keys like 'original_image' to actual image paths to be processed.
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
        response_text = f"VLM OCR result: {result}"
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


class ExtractTextFromImageWithVLM(BaseMineruVLMOCRTool):
    """Extracts text from an image using MinerU VLM OCR."""

    async def _execute_logic(self, instance_id: str, parameters: dict[str, Any]) -> str:
        # Initialize VLM resources if needed
        if _vlm_semaphore is None:
            _initialize_vlm_resources()

        image_key = parameters["image"]
        # server_url = parameters.get("server_url", "http://127.0.0.1:30000")
        server_url = MINERU_SERVER_URL
        image_path = _resolve_from_store(self._instance_dict[instance_id], image_key)

        logger.info(f"VLM OCR tool called for image: {image_key}")

        # Use semaphore to limit concurrent VLM requests
        async with _vlm_semaphore:
            try:
                result = await _run_vlm_inference(
                    image_path, server_url, DEFAULT_RETURN_MODE
                )
                logger.info(f"VLM OCR completed for {image_key}")
                return result
            except Exception as e:
                error_msg = f"Error processing image {image_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg
