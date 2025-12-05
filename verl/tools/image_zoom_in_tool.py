import logging
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

MIN_QWEN_DIMENSION = 28


def _resolve_from_store(instance_store: dict, image_key: str) -> str:
    """Resolve an image key like 'original_image' from the per-instance response_store."""
    imgs_map: dict | None = instance_store.get("response_store", {}).get("imgs_map", {})

    if imgs_map and isinstance(imgs_map, dict) and image_key in imgs_map:
        return str(imgs_map[image_key])

    error_msg = (
        f"Cannot resolve image key '{image_key}'. "
        f"Current images you have are: {list(imgs_map.keys()) if imgs_map else []}. "
        "You can call this tool like this: "
        '<tool_call>\n{{"name": "image_zoom_in_tool", '
        '"arguments": {{"image": "original_image", "bbox_2d": [100, 100, 900, 900]}}}}\n</tool_call>'
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


class ImageZoomInTool(BaseTool):
    """A tool for zooming in on an image by cropping it based on a bounding box."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

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

    def _validate_bbox(
        self, left: float, top: float, right: float, bottom: float
    ) -> bool:
        """Validate the bounding box dimensions and aspect ratio."""
        try:
            if not (left < right and top < bottom):
                return False
            height = bottom - top
            width = right - left
            if min(height, width) == 0:
                return False
            # Prevent extreme aspect ratios which are likely errors
            if max(height, width) / min(height, width) > 100:
                return False
            return True
        except Exception as e:
            logger.warning(f"Bbox validation error: {e}")
            return False

    def _maybe_resize_absolute_bbox(
        self, bbox_abs: list[float], image_width: int, image_height: int
    ) -> Optional[list[int]]:
        """
        Clamps, validates, and potentially resizes a bounding box using absolute pixel coordinates.
        If the initial box is smaller than MIN_QWEN_DIMENSION, it expands from the center,
        shifting the box to stay within bounds. A final validation ensures the output is valid.
        """
        left, top, right, bottom = bbox_abs

        # 1. Clamp the initial bounding box to the image dimensions.
        left = max(0.0, left)
        top = max(0.0, top)
        right = min(float(image_width), right)
        bottom = min(float(image_height), bottom)

        if not self._validate_bbox(left, top, right, bottom):
            return None

        height, width = bottom - top, right - left

        # 2. If the box is too small, attempt to resize it.
        if height < MIN_QWEN_DIMENSION or width < MIN_QWEN_DIMENSION:
            center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
            min_dim = min(height, width)
            if min_dim == 0:
                return None

            ratio = MIN_QWEN_DIMENSION / min_dim
            target_width, target_height = width * ratio, height * ratio

            # Scale down to fit if the target size is larger than the image.
            if target_width > image_width:
                scale_down = image_width / target_width
                target_width, target_height = image_width, target_height * scale_down
            if target_height > image_height:
                scale_down = image_height / target_height
                target_height, target_width = image_height, target_width * scale_down

            # Determine new coordinates and shift the box to stay within image boundaries.
            new_left = center_x - target_width / 2.0
            new_top = center_y - target_height / 2.0
            if new_left < 0:
                new_left = 0
            if new_top < 0:
                new_top = 0
            if new_left + target_width > image_width:
                new_left = image_width - target_width
            if new_top + target_height > image_height:
                new_top = image_height - target_height

            left, top = new_left, new_top
            right, bottom = new_left + target_width, new_top + target_height

        # 3. Final validation and conversion to integer coordinates.
        final_bbox = [floor(left), floor(top), ceil(right), ceil(bottom)]
        final_left, final_top, final_right, final_bottom = final_bbox

        if not self._validate_bbox(final_left, final_top, final_right, final_bottom):
            return None

        if (final_bottom - final_top) < MIN_QWEN_DIMENSION or (
            final_right - final_left
        ) < MIN_QWEN_DIMENSION:
            return None

        return final_bbox

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        image_key = parameters.get("image")
        bbox_2d = parameters.get("bbox_2d")

        # --- Step 1: Retain your fine-grained validation on relative [0, 1000] coordinates ---
        if not image_key:
            return (
                ToolResponse(text="Error: image parameter is missing."),
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

        # --- Step 2: Core processing logic ---
        instance_data = self._instance_dict[instance_id]
        try:
            image_path = _resolve_from_store(instance_data, image_key)
            image = fetch_image({"image": image_path})
            image_width, image_height = image.size

            # Convert relative [0, 1000] coordinates to absolute pixel coordinates
            abs_x1 = rel_x1 / 1000.0 * image_width
            abs_y1 = rel_y1 / 1000.0 * image_height
            abs_x2 = rel_x2 / 1000.0 * image_width
            abs_y2 = rel_y2 / 1000.0 * image_height

            # Use the upgraded, robust function to process absolute pixel coordinates
            resized_bbox = self._maybe_resize_absolute_bbox(
                [abs_x1, abs_y1, abs_x2, abs_y2],
                image_width=image_width,
                image_height=image_height,
            )

            if resized_bbox is None:
                # Retain your fine-grained failure message
                return (
                    ToolResponse(
                        text=(
                            f"Error: crop failed after adjustment, original bbox={bbox_2d}. "
                            f"The specified region is invalid or results in a crop smaller than {MIN_QWEN_DIMENSION}x{MIN_QWEN_DIMENSION} pixels. "
                            f"Please provide a larger valid region."
                        )
                    ),
                    -0.05,
                    {"success": False},
                )

            # Crop the image directly. No smart_resize is needed.
            cropped_image = image.crop(resized_bbox)
            logger.info(f"Cropped image to size: {cropped_image.size}")

            # --- Optional: auto-enlarge small crops for better visibility ---
            width, height = cropped_image.size
            if min(width, height) < 224:
                scale_factor = 2.0
                new_size = (int(width * scale_factor), int(height * scale_factor))
                cropped_image = cropped_image.resize(new_size, Image.BICUBIC)
                logger.info(f"Upscaled cropped image to {new_size}")

        except Exception as e:
            logger.error(f"Error processing image zoom-in: {e}", exc_info=True)
            return (
                ToolResponse(text=f"Error processing image zoom-in: {e}"),
                -0.05,
                {"success": False},
            )

        # --- Step 3: Retain your custom prompt generation logic ---
        action_x = len(
            self._instance_dict[instance_id]["response_store"].get("imgs_map", {})
        ) - 1
        response_text = (
            f"This is the zoom-in image `observation_{action_x + 1}`(zoomed in on the image {image_key}) after your tool call.\n"
            "Continue your reasoning process within `<think>...</think>`. "
            "If needed, you can continue to call available tools within `<tool_call>...</tool_call>`. "
            "If your final response to the original question is confirmed, put your final answer or judgment within `<answer>...</answer>`."
        )

        return (
            ToolResponse(image=[cropped_image], text=response_text),
            0.0,
            {"success": True},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
