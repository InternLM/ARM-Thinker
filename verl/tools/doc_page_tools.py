# -*- coding: utf-8 -*-
import os

import logging
import os
import asyncio
from typing import Any, Optional, Dict, List
from uuid import uuid4
from PIL import Image, ImageFile

from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

import warnings

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction
from chromadb.config import Settings

# ============== Pillow Safety Settings ==============
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============== Logging and Environment ==============
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

RETRIEVE_K = int(os.getenv("RETRIEVE_K", "5"))
IMAGE_ROOT = os.getenv("RAG_IMAGE_ROOT", None)
DB_PATH = os.getenv("DB_PATH", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multi_pdf_clip_collection")
RAG_IMAGE_MAX_SIDE = int(os.getenv("RAG_IMAGE_MAX_SIDE", "1120"))
MAX_CONCAT_PIXELS = int(os.getenv("MAX_CONCAT_PIXELS", str(80_000_000)))
SUPPORTED_IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


# ============== Embedding Function ==============
class MultiModalCLIPEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = SentenceTransformer(
                model_name,
                device=device,
            )
            self.device = device
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model '{model_name}': {e}"
            )
            raise

    def __call__(self, texts: Documents) -> List[List[float]]:
        text_embs = self._model.encode(
            texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False
        )
        return text_embs.tolist()


# ============== RetrieverManager ==============
class RetrieverManager:
    def __init__(self):
        if hasattr(self, "collection") and self.collection is not None:
            return
        self.collection = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        if self.collection is not None:
            return
        async with self._lock:
            if self.collection is not None:
                return
            logger.info("Initializing RetrieverManager...")
            try:
                embedding_function = MultiModalCLIPEmbeddingFunction()
                chroma_client = chromadb.PersistentClient(
                    path=DB_PATH, settings=Settings(anonymized_telemetry=False)
                )
                self.collection = chroma_client.get_collection(
                    name=COLLECTION_NAME, embedding_function=embedding_function
                )
                logger.info("RetrieverManager ready.")
            except Exception as e:
                logger.error(f"RetrieverManager initialization failed: {e}")
                self.collection = None


# Module-level global instance (no singleton trickery)
retriever_manager = RetrieverManager()

# ============== Common Image Functions ==============
def _resize_keep_long_side(im: Image.Image, target_side: int) -> Image.Image:
    w, h = im.size
    max_side = max(w, h)
    if max_side <= target_side:
        return im
    if w >= h:
        new_w, new_h = target_side, int(target_side * h / w)
    else:
        new_h, new_w = target_side, int(target_side * w / h)
    return im.resize((new_w, new_h), Image.BICUBIC)


def load_and_resize_image(
    image_path: str, target_side: int = RAG_IMAGE_MAX_SIDE
) -> Image.Image:
    """Read a single image and proportionally resize it so the longest side equals target_side."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    im = Image.open(image_path).convert("RGB")
    return _resize_keep_long_side(im, target_side)


def concat_images_horizontally(
    image_paths: List[str], max_per_row: int = 3, target_side: int = 1120, pad: int = 10
) -> Image.Image:
    """Horizontally stitch up to max_per_row images; each image is first resized so its longest side equals target_side."""
    imgs = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]
    if not imgs:
        raise FileNotFoundError("No valid image paths found for concatenation.")

    imgs = imgs[:max_per_row]
    resized = [_resize_keep_long_side(im, target_side) for im in imgs]

    total_w = sum(im.width for im in resized) + pad * (len(resized) - 1)
    max_h = max(im.height for im in resized)

    stitched = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in resized:
        stitched.paste(im, (x, (max_h - im.height) // 2))
        x += im.width + pad

    # Safely downsample to limit total pixel count
    if stitched.width * stitched.height > MAX_CONCAT_PIXELS:
        scale = (MAX_CONCAT_PIXELS / (stitched.width * stitched.height)) ** 0.5
        new_size = (int(stitched.width * scale), int(stitched.height * scale))
        stitched = stitched.resize(new_size, Image.BICUBIC)
        logger.warning(
            f"Downscaled concatenated image to {new_size} due to size limit."
        )
    return stitched


def find_existing_image_path(filename: str, page: int) -> str:
    """Try common extensions in order for {filename}_{page}.{ext}; return the existing path, otherwise return the preferred .jpg path (which may not exist)."""
    for ext in SUPPORTED_IMG_EXTS:
        candidate = os.path.join(IMAGE_ROOT, f"{filename}_{int(page)}{ext}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(IMAGE_ROOT, f"{filename}_{int(page)}.jpg")


def _all_tried_paths(filename: str, page: int) -> List[str]:
    """Return all attempted paths for a given page (for error messages)."""
    return [
        os.path.join(IMAGE_ROOT, f"{filename}_{int(page)}{ext}")
        for ext in SUPPORTED_IMG_EXTS
    ]


# ============== Tool 1: DocPageSearchTool (Search by query and stitch) ==========================
class DocPageSearchTool(BaseTool):
    """
    Search pages in a document by natural-language query and return
    one image that concatenates the top-k matched pages.
    Parameters:
      - filename: str
      - query: str
      - top_k: int (optional, default from env RETRIEVE_K)
    """

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
        if instance_id is None:
            instance_id = str(uuid4())
        new_response_store = kwargs.get("create_kwargs", {}).get(
            "response_store", response_store or {}
        )
        self._instance_dict[instance_id] = {"response_store": new_response_store}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: Dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        await retriever_manager.initialize()
        if retriever_manager.collection is None:
            msg = "Error: RetrieverManager failed to initialize."
            logger.error(msg)
            return ToolResponse(text=msg), -0.05, {"success": False}

        filename = parameters.get("filename")
        query = parameters.get("query")

        if not filename:
            return (
                ToolResponse(text="Error: missing required parameter 'filename'."),
                -0.05,
                {"success": False},
            )
        if not query:
            return (
                ToolResponse(text="Error: missing required parameter 'query'."),
                -0.05,
                {"success": False},
            )

        try:
            results = retriever_manager.collection.query(
                query_texts=[query], n_results=RETRIEVE_K, where={"source": filename}
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return ToolResponse(text=f"Retrieval error: {e}"), -0.05, {"success": False}

        if not results or not results.get("ids") or not results["ids"][0]:
            # response_text = (
            #     f"No relevant content found for '{query}' in document '{filename}'. "
            #     f"Try a different query or finish without calling this tool."
            # )
            response_text = f"No relevant content found for '{query}' in document '{filename}'. Maybe you should check your provided filename in your tool calling."
            return ToolResponse(text=response_text), 0.0, {"success": True, "pages": []}

        try:
            # Deduplicate by page from metadata and extract the first k in order (as int)
            pages_in_order: List[int] = []
            for m in results["metadatas"][0]:
                page = m.get("page")
                if page is None:
                    continue
                try:
                    page_i = int(page)
                except (TypeError, ValueError):
                    continue
                if page_i not in pages_in_order:
                    pages_in_order.append(page_i)
                if len(pages_in_order) >= RETRIEVE_K:
                    break

            page_image_paths: List[str] = []
            for p in pages_in_order:
                img_path = find_existing_image_path(filename, p)
                if os.path.exists(img_path):
                    page_image_paths.append(img_path)
                else:
                    tried_paths = _all_tried_paths(filename, p)
                    raise FileNotFoundError(
                        "Image not found for page {} in {}. Tried: {}".format(
                            p, filename, " | ".join(tried_paths)
                        )
                    )

            stitched_image = await asyncio.to_thread(
                concat_images_horizontally,
                page_image_paths,
                max_per_row=RETRIEVE_K,
                target_side=RAG_IMAGE_MAX_SIDE,
                pad=8,
            )
            response_text = f"Concatenated page images for query '{query}' in '{filename}': pages {pages_in_order}. If the retrieved pages don't seem relevant, you can try using a different or more specific query string to improve the results."
            logger.info(response_text)
            return (
                ToolResponse(image=[stitched_image], text=response_text),
                0.0,
                {"success": True, "pages": pages_in_order},
            )

        except Exception as e:
            logger.error(f"Error in DocPageSearchTool: {e}")
            return ToolResponse(text=f"Error: {e}"), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


# ============== Tool 2: DocPageByIndexTool (Return a single page by index) ==========================
class DocPageByIndexTool(BaseTool):
    """
    Return a single page image by 1-based image/page index.
    Parameters:
      - filename: str
      - image_idx: int (>=1)
    """

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
        if instance_id is None:
            instance_id = str(uuid4())
        new_response_store = kwargs.get("create_kwargs", {}).get(
            "response_store", response_store or {}
        )
        self._instance_dict[instance_id] = {"response_store": new_response_store}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: Dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        # Does not rely on vector retrieval; no need to initialize retriever_manager

        filename = parameters.get("filename")
        image_idx = parameters.get("image_idx")

        if not filename:
            return (
                ToolResponse(text="Error: missing required parameter 'filename'."),
                -0.05,
                {"success": False},
            )
        if image_idx is None:
            return (
                ToolResponse(text="Error: missing required parameter 'image_idx'."),
                -0.05,
                {"success": False},
            )

        try:
            idx = int(image_idx)
        except Exception:
            return (
                ToolResponse(text="Error: 'image_idx' must be an integer (1-based)."),
                -0.05,
                {"success": False},
            )
        if idx < 1:
            return (
                ToolResponse(text="Error: 'image_idx' must be >= 1 (1-based index)."),
                -0.05,
                {"success": False},
            )

        try:
            img_path = find_existing_image_path(filename, idx)
            if not os.path.exists(img_path):
                tried_paths = _all_tried_paths(filename, idx)
                raise FileNotFoundError(
                    "Image not found for page {} in {}. Maybe you input the wrong filename or image index out of range. Please check your input and try again."
                )
            img = await asyncio.to_thread(
                load_and_resize_image, img_path, RAG_IMAGE_MAX_SIDE
            )
            response_text = f"Here is page image #{idx} for document '{filename}'."
            logger.info(response_text)
            return (
                ToolResponse(image=[img], text=response_text),
                0.0,
                {"success": True, "page": idx},
            )
        except Exception as e:
            logger.error(f"DocPageByIndexTool error: {e}")
            return ToolResponse(text=f"Error: {e}"), -0.05, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
