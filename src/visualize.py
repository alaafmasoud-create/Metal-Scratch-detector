from __future__ import annotations

import cv2
import numpy as np

from .detect_classic import DetectionResult


def overlay_mask_and_boxes(
    image_bgr: np.ndarray,
    result: DetectionResult,
    mask_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    output = image_bgr.copy()

    if result.mask is not None and np.count_nonzero(result.mask) > 0:
        color_layer = np.zeros_like(output)
        color_layer[result.mask > 0] = mask_color
        output = cv2.addWeighted(output, 1.0, color_layer, 0.35, 0)

    for (x, y, w, h) in result.boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), (20, 220, 20), 2)

    status_text = f"{result.label} | Confidence: {result.confidence:.1%}"
    cv2.rectangle(output, (10, 10), (min(output.shape[1] - 10, 520), 54), (0, 0, 0), -1)
    cv2.putText(
        output,
        status_text,
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def make_side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    left_h, left_w = left_bgr.shape[:2]
    right_h, right_w = right_bgr.shape[:2]

    target_h = max(left_h, right_h)

    def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == height:
            return img
        new_w = max(1, int(round(w * (height / float(h)))))
        return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)

    left = resize_to_height(left_bgr, target_h)
    right = resize_to_height(right_bgr, target_h)
    divider = np.full((target_h, 12, 3), 35, dtype=np.uint8)
    return np.hstack([left, divider, right])
