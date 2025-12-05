import os
import time
import uuid
import base64
from PIL import Image
from io import BytesIO


def generate_identifier_path_name():
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:12]
    return f"image_{ts}_{uid}"


def save_image_to_file(image: Image.Image, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.png")
    image.save(path)
    return path


def encode_image_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def encode_image_file(image_file: str) -> str:
    with open(image_file, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def decode_base64_image(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]  # remove prefix if exists
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))