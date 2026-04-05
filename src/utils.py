from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def image_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode image as PNG.")
    return buffer.tobytes()


def save_image(path: str | Path, image_bgr: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise ValueError(f"Failed to save image to {path}")
    return path


def pil_image_from_bgr(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(image_bgr))


def ensure_within_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def readable_filename(original_name: str, suffix: str = "_prediction.png") -> str:
    stem = Path(original_name).stem if original_name else "result"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return f"{safe}{suffix}"
