from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect_classic import DetectionResult, detect_scratches_classic
from .preprocess import preprocess_image


@dataclass
class InferenceOutput:
    result: DetectionResult
    resized_image_bgr: np.ndarray


def run_classic_inference(
    image_bgr: np.ndarray,
    max_dim: int = 1408,
    blur_kernel: int = 5,
    clip_limit: float = 1.5,
    threshold_bias: float = 1.65,
    min_area: int = 24,
) -> InferenceOutput:
    artifacts = preprocess_image(
        image_bgr=image_bgr,
        max_dim=max_dim,
        blur_kernel=blur_kernel,
        clip_limit=clip_limit,
    )
    result = detect_scratches_classic(
        artifacts=artifacts,
        threshold_bias=threshold_bias,
        min_area=min_area,
    )
    return InferenceOutput(result=result, resized_image_bgr=artifacts.resized_bgr)
