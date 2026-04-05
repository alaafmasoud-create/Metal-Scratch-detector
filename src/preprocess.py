from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessArtifacts:
    original_bgr: np.ndarray
    resized_bgr: np.ndarray
    gray: np.ndarray
    enhanced: np.ndarray
    scale: float


def resize_keep_aspect(image: np.ndarray, max_dim: int = 1408) -> tuple[np.ndarray, float]:
    """Resize while preserving aspect ratio. Returns (resized_image, scale)."""
    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_dim:
        return image.copy(), 1.0

    scale = max_dim / float(current_max)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def clahe_enhance(gray: np.ndarray, clip_limit: float = 2.5, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray)


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def preprocess_image(
    image_bgr: np.ndarray,
    max_dim: int = 1408,
    blur_kernel: int = 3,
    clip_limit: float = 2.5,
) -> PreprocessArtifacts:
    resized, scale = resize_keep_aspect(image_bgr, max_dim=max_dim)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = normalize_gray(gray)

    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    enhanced = clahe_enhance(gray, clip_limit=clip_limit, tile_grid_size=8)
    return PreprocessArtifacts(
        original_bgr=image_bgr,
        resized_bgr=resized,
        gray=gray,
        enhanced=enhanced,
        scale=scale,
    )


def safe_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)
