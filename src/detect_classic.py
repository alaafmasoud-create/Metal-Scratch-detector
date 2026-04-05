from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .preprocess import PreprocessArtifacts


@dataclass
class DetectionResult:
    label: str
    defect_detected: bool
    confidence: float
    score: float
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    debug_images: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _elongated_kernel(length: int, angle: str) -> np.ndarray:
    length = max(9, int(length))
    if length % 2 == 0:
        length += 1

    if angle == "horizontal":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (length, 3))
    if angle == "vertical":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (3, length))
    if angle == "diag1":
        kernel = np.zeros((length, length), dtype=np.uint8)
        cv2.line(kernel, (0, 0), (length - 1, length - 1), 1, 1)
        return kernel
    if angle == "diag2":
        kernel = np.zeros((length, length), dtype=np.uint8)
        cv2.line(kernel, (0, length - 1), (length - 1, 0), 1, 1)
        return kernel
    raise ValueError(f"Unknown angle: {angle}")


def _combine_blackhat(gray: np.ndarray) -> np.ndarray:
    responses = []
    for angle in ("horizontal", "vertical", "diag1", "diag2"):
        kernel = _elongated_kernel(25, angle)
        response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        responses.append(response)
    combined = np.maximum.reduce(responses)
    return combined


def _gradient_energy(gray: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def _clean_mask(mask: np.ndarray, min_area: int = 18) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    boxes: list[tuple[int, int, int, int]] = []

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue

        aspect = max(w, h) / max(1.0, min(w, h))
        fill_ratio = area / float(max(1, w * h))

        # Favor thin elongated regions typical of scratches.
        if aspect < 1.8 and area < 140:
            continue
        if fill_ratio > 0.95 and area < 250:
            continue

        cleaned[labels == idx] = 255
        boxes.append((x, y, w, h))

    if boxes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned, boxes


def _score_mask(mask: np.ndarray, boxes: list[tuple[int, int, int, int]], image_shape: tuple[int, int]) -> tuple[float, float]:
    h, w = image_shape
    img_area = float(max(1, h * w))
    mask_area = float(np.count_nonzero(mask))
    area_ratio = mask_area / img_area

    long_box_bonus = 0.0
    for _, _, bw, bh in boxes:
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        perimeter = 2.0 * (bw + bh)
        long_box_bonus += min(0.25, aspect / 40.0) + min(0.20, perimeter / 2500.0)

    raw_score = (area_ratio * 12.0) + long_box_bonus + min(0.3, len(boxes) * 0.03)
    confidence = float(1.0 / (1.0 + np.exp(-(raw_score - 0.18) * 7.0)))
    return float(raw_score), confidence


def detect_scratches_classic(
    artifacts: PreprocessArtifacts,
    threshold_bias: float = 1.15,
    min_area: int = 18,
) -> DetectionResult:
    gray = artifacts.enhanced
    blackhat = _combine_blackhat(gray)
    gradient = _gradient_energy(gray)

    fusion = cv2.addWeighted(blackhat, 0.75, gradient, 0.25, 0)
    fusion_blur = cv2.GaussianBlur(fusion, (3, 3), 0)

    base_thresh = max(8.0, float(np.mean(fusion_blur) + threshold_bias * np.std(fusion_blur)))
    _, binary = cv2.threshold(fusion_blur, base_thresh, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    cleaned, boxes = _clean_mask(closed, min_area=min_area)
    raw_score, confidence = _score_mask(cleaned, boxes, image_shape=gray.shape)
    defect_detected = confidence >= 0.58 and len(boxes) > 0

    label = "Defect detected" if defect_detected else "No defect"

    metadata = {
        "threshold": base_thresh,
        "candidates": len(boxes),
        "mask_pixels": int(np.count_nonzero(cleaned)),
        "scale": artifacts.scale,
    }

    return DetectionResult(
        label=label,
        defect_detected=defect_detected,
        confidence=float(confidence),
        score=float(raw_score),
        mask=cleaned,
        boxes=boxes,
        debug_images={
            "enhanced_gray": gray,
            "blackhat": blackhat,
            "gradient": gradient,
            "fusion": fusion_blur,
            "binary": binary,
            "cleaned_mask": cleaned,
        },
        metadata=metadata,
    )
