"""The 2x2 rule check.

The American Bird Conservancy states that markings must be no more than 2 inches apart
horizontally and vertically. A pane violates that wherever an axis-aligned 2 inch square fits
inside it containing no marking, which is exactly an erosion of empty space by a square of that
side. The rule is stated in horizontal and vertical terms, so the test is axis-aligned; a gap that
is only wide on the diagonal is not reported.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

RULE_MM = 50.8
MIN_KERNEL_PX = 3


@dataclass(frozen=True)
class RepairTarget:
    center_mm: tuple[float, float]
    area_mm2: float


@dataclass(frozen=True)
class CoverageResult:
    confirmed: np.ndarray
    possible: np.ndarray
    largest_gap_mm: float
    unknown_fraction: float
    targets: tuple[RepairTarget, ...]


def _erode_then_restore(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Border is treated as non-empty so that only squares lying fully inside the pane count.
    centers = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.dilate(centers, kernel)


def _square_fits(mask: np.ndarray, side: int) -> bool:
    kernel = np.ones((side, side), np.uint8)
    return bool(cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).any())


def _largest_empty_square_px(mask: np.ndarray) -> int:
    # Chessboard distance d at a pixel means a (2d - 1) px square centred there is all empty, and
    # no square of 2d + 1 px or more fits anywhere. One erosion settles the remaining 2d case.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    d = int(cv2.distanceTransform(padded, cv2.DIST_C, 3).max())
    if d == 0:
        return 0
    return 2 * d if _square_fits(mask, 2 * d) else 2 * d - 1


def check_coverage(pane, markers, unknown, mm_per_px, rule_mm=RULE_MM) -> CoverageResult:
    # Smallest whole-pixel square strictly wider than the rule, so a square of exactly 2 inches
    # passes, and the verdict flips at the same pixel count as the headline number below.
    side = math.floor(rule_mm / mm_per_px) + 1
    if side < MIN_KERNEL_PX:
        raise ValueError(
            f"{rule_mm} mm is {side} px at {mm_per_px} mm/px, too coarse to measure. "
            "Capture closer or rectify to a finer scale."
        )

    # OpenCV masks arrive as 0/255 uint8; bitwise logic and sums below assume booleans.
    pane, markers, unknown = (np.asarray(m, dtype=bool) for m in (pane, markers, unknown))
    pane_u8 = pane.astype(np.uint8)
    observed_empty = (pane & ~markers & ~unknown).astype(np.uint8)
    # Unknown is never counted as covered: a marker detected under glare cannot rule out a gap.
    possibly_empty = (pane & (~markers | unknown)).astype(np.uint8)
    kernel = np.ones((side, side), np.uint8)

    confirmed = _erode_then_restore(observed_empty, kernel).astype(bool) & pane
    any_violation = _erode_then_restore(possibly_empty, kernel).astype(bool) & pane
    possible = any_violation & ~confirmed

    # The headline is the same axis-aligned square the verdict tests, measured on observed pixels
    # only, so the number never disagrees with the verdict and unknown area never counts as a gap.
    # A largest-circle measure reads over 2 inches on a compliant 45 mm dot grid.
    largest_gap_mm = float(_largest_empty_square_px(observed_empty) * mm_per_px)

    pane_px = int(pane_u8.sum())
    unknown_fraction = float((unknown & pane).sum() / pane_px) if pane_px else 0.0

    px_area_mm2 = mm_per_px * mm_per_px
    count, _, stats, centroids = cv2.connectedComponentsWithStats(confirmed.astype(np.uint8), 8)
    targets = tuple(
        RepairTarget(
            center_mm=(float(centroids[i][0] * mm_per_px), float(centroids[i][1] * mm_per_px)),
            area_mm2=float(stats[i, cv2.CC_STAT_AREA] * px_area_mm2),
        )
        for i in range(1, count)
    )

    return CoverageResult(
        confirmed=confirmed,
        possible=possible,
        largest_gap_mm=largest_gap_mm,
        unknown_fraction=unknown_fraction,
        targets=targets,
    )
