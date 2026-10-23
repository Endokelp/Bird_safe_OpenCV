"""Detect visible bird-safe dots on a rectified pane."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

DOT_MM = 6.0
WORKING_MM_PER_PX = 0.5
# ponytail: 1.2 px at 0.5 mm/px. Stops texture from fragmenting adaptive threshold.
# Retune if real 6 mm dots smear. Held-out captures stay sealed.
PRE_BLUR_SIGMA = 1.2
TOPHAT_SCALE = 1.6
# A dot-sized block left flat dot interiors below the local mean, so big dots became rings that
# broke into C shapes. 1.5 found 31/31 real dots in session01; 2.0 added a false dot.
BLOCK_SCALE = 1.5
ADAPTIVE_C = -2
MIN_CIRCULARITY = 0.6
# ponytail: real 19.05 mm dots measured 18.4-19.1 mm (session01); 0.5 let 10 mm reflections in.
DIAMETER_LO = 0.7
DIAMETER_HI = 2.0
# Cuts thin bright bridges (tape edges, reflections) that merge a dot into a non-round blob.
OPEN_SCALE = 0.25


@dataclass(frozen=True)
class MarkerResult:
    mask: np.ndarray
    centers_mm: tuple[tuple[float, float], ...]


def _odd_ksize(value: float) -> int:
    return max(3, int(round(value)) | 1)


def detect_markers(image, mm_per_px, expected_dot_mm=DOT_MM) -> MarkerResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = mm_per_px / WORKING_MM_PER_PX
    src_h, src_w = gray.shape[:2]
    work_h = max(1, round(src_h * scale))
    work_w = max(1, round(src_w * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    working = cv2.resize(gray, (work_w, work_h), interpolation=interpolation)
    working = cv2.GaussianBlur(
        working, (_odd_ksize(PRE_BLUR_SIGMA * 6), _odd_ksize(PRE_BLUR_SIGMA * 6)), PRE_BLUR_SIGMA
    )

    dot_px = expected_dot_mm / WORKING_MM_PER_PX
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_ksize(TOPHAT_SCALE * dot_px),) * 2)
    block = _odd_ksize(BLOCK_SCALE * dot_px)
    opener = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_ksize(OPEN_SCALE * dot_px),) * 2)
    # Each polarity is filtered on its own. On a dense grid the glass between bright dots is a
    # dark-hat response too, and OR-ing the masks merged every dot into one lattice.
    contours = []
    for morph in (cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT):
        response = cv2.morphologyEx(working, morph, se)
        binary = cv2.adaptiveThreshold(
            response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, ADAPTIVE_C
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opener)
        contours += cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    mask_work = np.zeros((work_h, work_w), np.uint8)
    centers = []
    lo_mm = DIAMETER_LO * expected_dot_mm
    hi_mm = DIAMETER_HI * expected_dot_mm
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 1 or perimeter < 1:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        diameter_mm = 2 * math.sqrt(area / math.pi) * WORKING_MM_PER_PX
        if circularity < MIN_CIRCULARITY or not (lo_mm <= diameter_mm <= hi_mm):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        centers.append((cx * WORKING_MM_PER_PX, cy * WORKING_MM_PER_PX))
        cv2.drawContours(mask_work, [contour], -1, 255, -1)

    if (work_h, work_w) != (src_h, src_w):
        mask = cv2.resize(mask_work, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_work
    return MarkerResult(mask=mask.astype(bool), centers_mm=tuple(centers))
