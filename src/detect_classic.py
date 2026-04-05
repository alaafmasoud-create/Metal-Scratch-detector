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


def _clean_mask(mask: np.ndarray, min_area: int = 24) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    boxes: list[tuple[int, int, int, int]] = []

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue

        length = max(w, h)
        thickness = max(1, min(w, h))
        aspect = length / float(thickness)
        fill_ratio = area / float(max(1, w * h))

        if aspect < 2.8:
            continue
        if length < 18:
            continue
        if fill_ratio > 0.88 and area < 320:
            continue

        cleaned[labels == idx] = 255
        boxes.append((x, y, w, h))

    if boxes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned, boxes


def _strong_candidates(
    boxes: list[tuple[int, int, int, int]], image_shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    h, w = image_shape
    min_len = max(45, int(min(h, w) * 0.05))

    strong: list[tuple[int, int, int, int]] = []
    for x, y, bw, bh in boxes:
        length = max(bw, bh)
        thickness = max(1.0, min(bw, bh))
        aspect = length / thickness

        if aspect >= 4.0 and length >= min_len:
            strong.append((x, y, bw, bh))

    return strong


def _score_mask(mask: np.ndarray, boxes: list[tuple[int, int, int, int]], image_shape: tuple[int, int]) -> tuple[float, float]:
    h, w = image_shape
    img_area = float(max(1, h * w))
    mask_area = float(np.count_nonzero(mask))
    area_ratio = mask_area / img_area

    if not boxes:
        return 0.0, 0.0

    strong_boxes = _strong_candidates(boxes, image_shape)

    if not strong_boxes:
        raw_score = area_ratio * 1.2
        confidence = float(min(0.35, max(0.01, raw_score * 10.0)))
        return float(raw_score), confidence

    strong_count = len(strong_boxes)
    strong_ratio = strong_count / max(1, len(boxes))
    largest_len_ratio = max(max(bw, bh) for _, _, bw, bh in strong_boxes) / max(1.0, min(h, w))

    raw_score = (area_ratio * 1.5) + (strong_ratio * 1.1) + (largest_len_ratio * 1.6)
    confidence = float(1.0 / (1.0 + np.exp(-(raw_score - 0.9) * 4.0)))

    return float(raw_score), confidence


def detect_scratches_classic(
    artifacts: PreprocessArtifacts,
    threshold_bias: float = 1.65,
    min_area: int = 24,
) -> DetectionResult:
    gray = artifacts.enhanced
    blackhat = _combine_blackhat(gray)
    gradient = _gradient_energy(gray)

    fusion = cv2.addWeighted(blackhat, 0.78, gradient, 0.22, 0)
    fusion_blur = cv2.GaussianBlur(fusion, (5, 5), 0)

    base_thresh = max(10.0, float(np.mean(fusion_blur) + threshold_bias * np.std(fusion_blur)))
    _, binary = cv2.threshold(fusion_blur, base_thresh, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    margin = max(8, int(min(gray.shape) * 0.02))
    closed[:margin, :] = 0
    closed[-margin:, :] = 0
    closed[:, :margin] = 0
    closed[:, -margin:] = 0

    cleaned, boxes = _clean_mask(closed, min_area=min_area)
    raw_score, confidence = _score_mask(cleaned, boxes, image_shape=gray.shape)

    strong_candidates = _strong_candidates(boxes, gray.shape)
    mask_ratio = np.count_nonzero(cleaned) / float(cleaned.shape[0] * cleaned.shape[1])

    defect_detected = (
        confidence >= 0.62
        and len(strong_candidates) >= 1
        and mask_ratio < 0.08
    )

    label = "Defect detected" if defect_detected else "No defect"

    metadata = {
        "threshold": base_thresh,
        "candidates": len(boxes),
        "strong_candidates": len(strong_candidates),
        "mask_pixels": int(np.count_nonzero(cleaned)),
        "mask_ratio": float(mask_ratio),
        "scale": artifacts.scale,
    }

    return DetectionResult(
        label=label,
        defect_detected=defect_detected,
        confidence=float(confidence),
        score=float(raw_score),
        mask=cleaned,
        boxes=strong_candidates if defect_detected else boxes,
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
