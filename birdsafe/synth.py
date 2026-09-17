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


def render_pane(
    pane: SyntheticPane,
    *,
    dark_on_light: bool = True,
    light: int = 220,
    dark: int = 40,
    texture_std: float = 8.0,
    glare_boxes_mm: tuple = (),
    blur_sigma_px: float = 0.0,
    rng_seed: int = 0,
) -> np.ndarray:
    height, width = pane.markers.shape
    background, foreground = (light, dark) if dark_on_light else (dark, light)
    gray = np.full((height, width), background, np.uint8)
    if texture_std > 0:
        noise = np.random.default_rng(rng_seed).normal(0.0, texture_std, (height, width))
        gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gray[pane.markers] = foreground
    for x_mm, y_mm, box_w_mm, box_h_mm in glare_boxes_mm:
        x0 = max(0, round(x_mm / pane.mm_per_px))
        y0 = max(0, round(y_mm / pane.mm_per_px))
        x1 = min(width, round((x_mm + box_w_mm) / pane.mm_per_px))
        y1 = min(height, round((y_mm + box_h_mm) / pane.mm_per_px))
        gray[y0:y1, x0:x1] = 255
    if blur_sigma_px > 0:
        ksize = max(3, int(round(blur_sigma_px * 6)) | 1)
        gray = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma_px)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
