"""Unreadable pixels: glare and blur that hide the glass surface."""

from dataclasses import dataclass

import cv2
import numpy as np

from birdsafe.markers import DOT_MM

SATURATION = 250
# floors measured on synthetic texture_std=8 (sharp Laplacian energy ~3800,
# blurred ~4, glare interior std 0). Retune on development captures. Held-out stays sealed.
TEXTURE_STD = 8.0
BLUR_ENERGY_FLOOR = 200.0
WINDOW_SCALE = 2.0


@dataclass(frozen=True)
class QualityResult:
    unknown: np.ndarray
    glare: np.ndarray
    blur: np.ndarray


def unknown_regions(image, mm_per_px, expected_dot_mm=DOT_MM) -> QualityResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    window = max(3, int(round(WINDOW_SCALE * expected_dot_mm / mm_per_px)) | 1)
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (window, window))
    mean_sq = cv2.blur(gray_f * gray_f, (window, window))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    glare = (gray >= SATURATION) & (std < TEXTURE_STD)

    laplacian = cv2.Laplacian(gray_f, cv2.CV_64F, ksize=3)
    energy = cv2.blur(laplacian * laplacian, (window, window))
    blur = energy < BLUR_ENERGY_FLOOR
    unknown = glare | blur
    return QualityResult(unknown=unknown, glare=glare, blur=blur)
