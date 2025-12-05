import logging
import math
import os
from math import ceil, floor
from typing import Any, Optional, Dict
from uuid import uuid4

from qwen_vl_utils import fetch_image
from PIL import Image

from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _resolve_image_by_idx(instance_store: dict, img_idx: int) -> str:
    """Resolve an image by index from the per-instance response_store.
    
    Images are ordered by: original_image first, then observation_1, observation_2, etc.
    """
    imgs_map: dict | None = instance_store.get("response_store", {}).get("imgs_map", {})

    if not imgs_map or not isinstance(imgs_map, dict):
        error_msg = (
            f"Cannot resolve image at index {img_idx}. "
            f"No images found in response_store."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Sort keys: original_image first, then observation_1, observation_2, etc.
    def sort_key(key: str) -> tuple[int, int]:
        if key == "original_image":
            return (0, 0)
        elif key.startswith("observation_"):
            try:
                idx = int(key.split("_")[1])
                return (1, idx)
            except (ValueError, IndexError):
                return (2, 0)  # Unknown observation format
        else:
            return (2, 0)  # Other keys come last

    sorted_keys = sorted(imgs_map.keys(), key=sort_key)
    
    if img_idx < 0 or img_idx >= len(sorted_keys):
        error_msg = (
            f"Cannot resolve image at index {img_idx}. "
            f"Available indices: 0 to {len(sorted_keys) - 1}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    image_key = sorted_keys[img_idx]
    return str(imgs_map[image_key])


class ImageZoomInToolQwen3VL(BaseTool):
    """A tool for zooming in on an image by cropping it based on a bounding box.
    
    This tool uses Qwen3-VL's official crop logic with smart_resize.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the tool schema in OpenAI format."""
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        response_store: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        new_response_store = kwargs.get("create_kwargs", {}).get(
            "response_store", response_store or {}
        )
        self._instance_dict[instance_id] = {"response_store": new_response_store}
        return instance_id, ToolResponse()

    # Image resizing functions (copied from Qwen3-VL official code)
    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 32,
        min_pixels: int = 56 * 56,
        max_pixels: int = 12845056,
    ) -> tuple[int, int]:
        """Smart resize image dimensions based on factor and pixel constraints"""
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def maybe_resize_bbox(
        self, left: float, top: float, right: float, bottom: float, img_width: int, img_height: int
    ) -> list[int]:
        """Resize bbox to ensure it's valid (copied from Qwen3-VL official code)"""
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        height = bottom - top
        width = right - left
        if height < 32 or width < 32:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 32 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)

            # Ensure the resized bbox is within image bounds
            new_left = max(0, new_left)
            new_top = max(0, new_top)
            new_right = min(img_width, new_right)
            new_bottom = min(img_height, new_bottom)

            new_height = new_bottom - new_top
            new_width = new_right - new_left

            if new_height > 32 and new_width > 32:
                return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        img_idx = parameters.get("img_idx")
        bbox_2d = parameters.get("bbox_2d")
        label = parameters.get("label", "")  # Optional label parameter

        # --- Step 1: Validate parameters ---
        if img_idx is None:
            return (
                ToolResponse(text="Error: img_idx parameter is missing."),
                -0.05,
                {"success": False},
            )
        try:
            img_idx = int(img_idx)
        except (ValueError, TypeError):
            return (
                ToolResponse(
                    text=f"Error: img_idx must be an integer, got {img_idx}."
                ),
                -0.05,
                {"success": False},
            )

        if not bbox_2d or len(bbox_2d) != 4:
            return (
                ToolResponse(
                    text=f"Error: bbox_2d must be a list of 4 numbers, got {bbox_2d}."
                ),
                -0.05,
                {"success": False},
            )
        try:
            rel_x1, rel_y1, rel_x2, rel_y2 = map(float, bbox_2d)
        except (ValueError, TypeError):
            return (
                ToolResponse(
                    text=f"Error: bbox_2d contains non-numeric values: {bbox_2d}."
                ),
                -0.05,
                {"success": False},
            )
        for i, v in enumerate([rel_x1, rel_y1, rel_x2, rel_y2]):
            if not (0 <= v <= 1000):
                return (
                    ToolResponse(
                        text=f"Error: bbox_2d[{i}]={v} out of range [0,1000], full bbox={bbox_2d}."
                    ),
                    -0.05,
                    {"success": False},
                )
        if rel_x1 >= rel_x2 or rel_y1 >= rel_y2:
            return (
                ToolResponse(
                    text=f"Error: bbox_2d must satisfy x1<x2 and y1<y2, got {bbox_2d}."
                ),
                -0.05,
                {"success": False},
            )

        # --- Step 2: Core processing logic using Qwen3-VL's crop logic ---
        instance_data = self._instance_dict[instance_id]
        try:
            image_path = _resolve_image_by_idx(instance_data, img_idx)
            image = fetch_image({"image": image_path})
            image_width, image_height = image.size

            # Convert relative [0, 1000] coordinates to absolute pixel coordinates
            abs_x1 = rel_x1 / 1000.0 * image_width
            abs_y1 = rel_y1 / 1000.0 * image_height
            abs_x2 = rel_x2 / 1000.0 * image_width
            abs_y2 = rel_y2 / 1000.0 * image_height

            # Use Qwen3-VL's maybe_resize_bbox to validate and resize bbox
            validated_bbox = self.maybe_resize_bbox(
                abs_x1, abs_y1, abs_x2, abs_y2, image_width, image_height
            )

            left, top, right, bottom = validated_bbox

            # Crop the image
            cropped_image = image.crop((left, top, right, bottom))
            logger.info(f"Cropped image to size: {cropped_image.size}")

            # Resize according to smart_resize logic
            # Note: Following official Qwen3-VL code exactly - smart_resize is called with (width, height)
            # even though function signature is (height, width), this matches official implementation
            new_w, new_h = self.smart_resize(
                (right - left), (bottom - top), factor=32, min_pixels=256 * 32 * 32
            )
            cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
            logger.info(f"Resized cropped image to {new_w}x{new_h} using smart_resize")

        except ValueError as e:
            # Handle image resolution errors
            return (
                ToolResponse(text=str(e)),
                -0.05,
                {"success": False},
            )
        except Exception as e:
            logger.error(f"Error processing image zoom-in: {e}", exc_info=True)
            return (
                ToolResponse(text=f"Error processing image zoom-in: {e}"),
                -0.05,
                {"success": False},
            )

        # --- Step 3: Generate response text ---
        # Get the index of the returned image (current length of imgs_map, as the new image will be added)
        current_img_count = len(
            self._instance_dict[instance_id]["response_store"].get("imgs_map", {})
        )
        response_text = (
            f"This is the zoom-in image from image at index {img_idx}. "
            f"The returned image is at index {current_img_count}."
        )

        return (
            ToolResponse(image=[cropped_image], text=response_text),
            0.0,
            {"success": True},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

