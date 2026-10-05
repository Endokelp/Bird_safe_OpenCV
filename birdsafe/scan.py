"""One step of a scan: fold a capture into the pane's evidence, then decide what to do next."""

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from birdsafe.calib import Rectified, Rejected
from birdsafe.coverage import check_coverage
from birdsafe.fusion import Observation, fuse
from birdsafe.markers import detect_markers
from birdsafe.policy import CaptureMetrics, TraceRecord, capture_rule, record
from birdsafe.quality import unknown_regions
from tools.charuco_sheet import sheet_footprint_mm


@dataclass(frozen=True)
class ScanState:
    canvas_mm: tuple[float, float, float, float]
    mm_per_px: float = 0.5
    observations: tuple[Observation, ...] = ()
    trace: tuple[TraceRecord, ...] = ()


def _capture_fields(capture: Rectified | Rejected) -> dict:
    if isinstance(capture, Rejected):
        return {
            "reject_reason": capture.reason,
            "reproj_rms_mm": None,
            "view_angle_deg": None,
            "native_mm_per_px": None,
        }
    return {
        "reject_reason": None,
        "reproj_rms_mm": capture.reproj_rms_mm,
        "view_angle_deg": capture.view_angle_deg,
        "native_mm_per_px": capture.native_mm_per_px,
    }


def _observe(capture: Rectified) -> Observation:
    markers = detect_markers(capture.image, capture.mm_per_px)
    quality = unknown_regions(capture.image, capture.mm_per_px)
    return Observation(
        markers.mask, quality.unknown | ~capture.valid, capture.board_origin_mm, capture.mm_per_px
    )


def _region(state: ScanState, shape: tuple[int, int]) -> np.ndarray:
    # The calibration sheet is taped on the glass in every frame. Treating it as unknown would ask
    # for recaptures that can never succeed, so it is left out and the pass message says so.
    x_mm, y_mm = state.canvas_mm[:2]
    x0, y0, x1, y1 = sheet_footprint_mm()
    scale = state.mm_per_px
    left, top = (max(0, math.floor((v - o) / scale)) for v, o in ((x0, x_mm), (y0, y_mm)))
    right, bottom = (max(0, math.ceil((v - o) / scale)) for v, o in ((x1, x_mm), (y1, y_mm)))
    region = np.ones(shape, bool)
    region[top:bottom, left:right] = False
    return region


def _centroids_mm(mask: np.ndarray, mm_per_px: float) -> tuple[tuple[float, float], ...]:
    """Centroids of the 8-connected components of mask, largest first, from the canvas top-left."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    by_area = sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return tuple(
        (float(centroids[i][0] * mm_per_px), float(centroids[i][1] * mm_per_px)) for i in by_area
    )


def _fused_fields(state: ScanState, observations: tuple[Observation, ...]) -> dict:
    fused = fuse(observations, state.canvas_mm, mm_per_px=state.mm_per_px)
    region = _region(state, fused.unknown.shape)
    coverage = check_coverage(region, fused.markers, fused.unknown, state.mm_per_px)
    unknown_centroids = _centroids_mm(fused.unknown & region, state.mm_per_px)
    return {
        "frames": fused.frames,
        "unknown_fraction": coverage.unknown_fraction,
        "unknown_centroid_mm": unknown_centroids[0] if unknown_centroids else None,
        "possible_centroids_mm": _centroids_mm(coverage.possible, state.mm_per_px),
        "repair_targets": tuple((*t.center_mm, t.area_mm2) for t in coverage.targets),
        "largest_gap_mm": coverage.largest_gap_mm,
    }


def scan_step(state: ScanState, capture: Rectified | Rejected) -> ScanState:
    observations = state.observations
    # A repair physically changed the pane, so frames taken before it describe glass that is gone.
    if state.trace and state.trace[-1].decision.action == "repair":
        observations = ()

    # capture_rule reads only the capture's own fields, so the fused placeholders never matter here.
    probe = CaptureMetrics(
        **_capture_fields(capture),
        frames=0,
        unknown_fraction=0.0,
        unknown_centroid_mm=None,
        possible_centroids_mm=(),
        repair_targets=(),
        largest_gap_mm=0.0,
    )
    if capture_rule(probe) is None:
        observations += (_observe(capture),)
    metrics = replace(probe, **_fused_fields(state, observations))

    return replace(
        state, observations=observations, trace=(*state.trace, record(metrics, state.trace))
    )
