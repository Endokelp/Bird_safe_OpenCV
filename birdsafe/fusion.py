"""Fuse several rectified captures of one pane onto a shared canvas in the board frame."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    markers: np.ndarray
    unknown: np.ndarray
    board_origin_mm: tuple[float, float]
    mm_per_px: float


@dataclass(frozen=True)
class FusedMasks:
    markers: np.ndarray
    unknown: np.ndarray
    frames: int


def fuse(
    observations: Sequence[Observation],
    canvas_mm: tuple[float, float, float, float],
    *,
    mm_per_px: float | None = None,
) -> FusedMasks:
    scales = {o.mm_per_px for o in observations}
    if mm_per_px is not None:
        scales.add(mm_per_px)
    if not scales:
        raise ValueError("mm_per_px is required when there are no observations to take it from.")
    if len(scales) > 1:
        raise ValueError(f"Observations disagree on mm_per_px: {sorted(scales)}.")
    (scale,) = scales

    x_mm, y_mm, width_mm, height_mm = canvas_mm
    height, width = round(height_mm / scale), round(width_mm / scale)
    markers = np.zeros((height, width), bool)
    resolved_by_any = np.zeros((height, width), bool)

    for obs in observations:
        # Whole-pixel placement is off by at most half a pixel, under 0.25 mm at 0.5 mm/px.
        shift_x = round((x_mm + obs.board_origin_mm[0]) / scale)
        shift_y = round((y_mm + obs.board_origin_mm[1]) / scale)
        obs_height, obs_width = obs.unknown.shape
        x0, x1 = max(0, -shift_x), min(width, obs_width - shift_x)
        y0, y1 = max(0, -shift_y), min(height, obs_height - shift_y)
        if x1 <= x0 or y1 <= y0:
            continue
        on_canvas = (slice(y0, y1), slice(x0, x1))
        on_frame = (slice(y0 + shift_y, y1 + shift_y), slice(x0 + shift_x, x1 + shift_x))

        # OpenCV masks arrive as 0/255 uint8; the logic below assumes booleans.
        resolved = ~np.asarray(obs.unknown, dtype=bool)[on_frame]
        # A marker detected under glare must not hide a gap that another frame resolved as empty,
        # so a frame contributes markers only where it could actually see.
        markers[on_canvas] |= np.asarray(obs.markers, dtype=bool)[on_frame] & resolved
        resolved_by_any[on_canvas] |= resolved

    return FusedMasks(markers=markers, unknown=~resolved_by_any, frames=len(observations))
