"""Synthetic panes with known layouts. Every artifact made from these is labelled synthetic."""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticPane:
    markers: np.ndarray
    pane: np.ndarray
    unknown: np.ndarray
    mm_per_px: float
    planted_gap_centers_mm: tuple[tuple[float, float], ...]


def grid_positions(extent_mm: float, spacing_mm: float) -> list[float]:
    """Dot positions that tile the whole pane at exactly the nominal spacing.

    Counting with floor leaves an untreated band at the far edge that can exceed the rule
    threshold, which would make a nominally compliant fixture fail for a reason unrelated to
    the spacing under test. Rounding the count up and centring the grid keeps each margin
    below half a spacing.
    """
    n = max(1, math.ceil(extent_mm / spacing_mm))
    inset = (extent_mm - (n - 1) * spacing_mm) / 2.0
    return [inset + i * spacing_mm for i in range(n)]


def make_pane(width_mm, height_mm, spacing_mm, dot_mm, mm_per_px,
              omit=(), unknown_boxes_mm=()) -> SyntheticPane:
    w = round(width_mm / mm_per_px)
    h = round(height_mm / mm_per_px)
    markers = np.zeros((h, w), np.uint8)
    unknown = np.zeros((h, w), np.uint8)
    radius = max(1, round(dot_mm / 2 / mm_per_px))
    omitted = set(omit)
    gaps = []

    xs = grid_positions(width_mm, spacing_mm)
    ys = grid_positions(height_mm, spacing_mm)
    for row, y_mm in enumerate(ys):
        for col, x_mm in enumerate(xs):
            if (row, col) in omitted:
                gaps.append((x_mm, y_mm))
                continue
            cv2.circle(markers, (round(x_mm / mm_per_px), round(y_mm / mm_per_px)), radius, 255, -1)

    for x_mm, y_mm, bw_mm, bh_mm in unknown_boxes_mm:
        x0, y0 = round(x_mm / mm_per_px), round(y_mm / mm_per_px)
        x1, y1 = round((x_mm + bw_mm) / mm_per_px), round((y_mm + bh_mm) / mm_per_px)
        unknown[y0:y1, x0:x1] = 255

    return SyntheticPane(
        markers=markers.astype(bool),
        pane=np.ones((h, w), bool),
        unknown=unknown.astype(bool),
        mm_per_px=mm_per_px,
        planted_gap_centers_mm=tuple(gaps),
    )
