"""Draw a scan step on its capture: found dots, unreadable glass, gaps, and the decision."""

import cv2
import numpy as np

from birdsafe.calib import Rectified
from birdsafe.coverage import RULE_MM, check_coverage
from birdsafe.fusion import fuse
from birdsafe.markers import detect_markers
from birdsafe.scan import ScanState, _region
from tools.charuco_sheet import sheet_footprint_mm

GREEN, MAGENTA, RED, ORANGE, CYAN = (0, 220, 0), (255, 0, 255), (0, 0, 255), (0, 165, 255), (255, 255, 0)
BANNER = {"pass": (40, 150, 40), "repair": (30, 30, 200), "recapture": (0, 140, 230)}
# the area under test is the dot patch beside the sheet: the found dots' outer edges
# plus this margin. Bare glass past the outer dots is not part of a patch, and a missing corner
# dot shrinks it. A pane outline drawn on the phone replaces it once the UI has one.
MARGIN_MM = 3


def patch_canvas(capture: Rectified, dot_mm: float) -> tuple[float, float, float, float] | None:
    """Board-frame rectangle around the largest group of dots beside the sheet, or None."""
    scale = capture.mm_per_px
    mask = detect_markers(capture.image, scale, dot_mm).mask.astype(np.uint8)
    ox, oy = capture.board_origin_mm
    x0, y0, x1, y1 = sheet_footprint_mm()
    mask[max(0, round((y0 + oy) / scale)):max(0, round((y1 + oy) / scale)),
         max(0, round((x0 + ox) / scale)):max(0, round((x1 + ox) / scale))] = 0
    # Dots within three rule lengths of each other join one group, so a grid too sparse for the
    # rule is still measured as one area and fails. A stray blob farther out than that must not
    # stretch the area under test over bare glass.
    # a fixed reach; a pattern with gaps over 150 mm splits and checks only its largest
    # part. Ask the user to mark the pane outline if that case shows up in the field.
    reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(3 * RULE_MM / scale) | 1,) * 2)
    count, groups = cv2.connectedComponents(cv2.dilate(mask, reach))
    if count < 2:
        return None
    sizes = np.bincount(groups[mask > 0], minlength=count)
    sizes[0] = 0
    ys, xs = np.nonzero((groups == sizes.argmax()) & (mask > 0))
    left, top = float(xs.min() * scale - ox), float(ys.min() * scale - oy)
    right, bottom = float((xs.max() + 1) * scale - ox), float((ys.max() + 1) * scale - oy)
    return (left - MARGIN_MM, top - MARGIN_MM, right - left + 2 * MARGIN_MM, bottom - top + 2 * MARGIN_MM)


def _tint(image, mask, color, alpha=0.5):
    out = image.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def _place(mask: np.ndarray, shape: tuple[int, int], sx: int, sy: int) -> np.ndarray:
    """Paste a canvas mask into a capture-sized mask at (sx, sy), clipped to the capture."""
    out = np.zeros(shape, bool)
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, sx), max(0, sy), min(shape[1], sx + w), min(shape[0], sy + h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = mask[y0 - sy:y1 - sy, x0 - sx:x1 - sx]
    return out


def render_step(capture: Rectified, state: ScanState) -> tuple[np.ndarray, int]:
    """The annotated capture with a decision banner, and the number of dots drawn."""
    scale = state.mm_per_px
    fused = fuse(state.observations, state.canvas_mm, mm_per_px=scale)
    region = _region(state, fused.unknown.shape)
    coverage = check_coverage(region, fused.markers, fused.unknown, scale)

    # Canvas pixel (0, 0) sits here in the rectified capture, the same shift fuse() applies.
    sx = round((state.canvas_mm[0] + capture.board_origin_mm[0]) / scale)
    sy = round((state.canvas_mm[1] + capture.board_origin_mm[1]) / scale)
    h, w = fused.unknown.shape
    shape = capture.image.shape[:2]

    view = _tint(capture.image, _place(fused.unknown & region, shape, sx, sy), MAGENTA)
    view = _tint(view, _place(coverage.possible, shape, sx, sy), ORANGE)
    view = _tint(view, _place(coverage.confirmed, shape, sx, sy), RED, 0.55)
    dots, _ = cv2.findContours(
        _place(fused.markers, shape, sx, sy).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(view, dots, -1, GREEN, 3)
    cv2.rectangle(view, (sx, sy), (sx + w, sy + h), CYAN, 2)
    record = state.trace[-1]
    for x_mm, y_mm, _area in record.metrics.repair_targets:
        cx, cy = round(sx + x_mm / scale), round(sy + y_mm / scale)
        cv2.drawMarker(view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 40, 3)

    m, d = record.metrics, record.decision
    lines = [f"{d.action.upper()}  ({d.rule})   largest gap {m.largest_gap_mm:.0f} mm (limit {RULE_MM})"]
    if m.reproj_rms_mm is not None:
        lines.append(
            f"dots found {len(dots)}   unreadable {m.unknown_fraction:.0%}   "
            f"board fit {m.reproj_rms_mm:.2f} mm   view {m.view_angle_deg:.0f} deg"
        )
    banner = np.full((80, view.shape[1], 3), BANNER.get(d.action, (90, 90, 90)), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(banner, text, (12, 32 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    legend = np.full((40, view.shape[1], 3), 30, np.uint8)
    x = 12
    for label, color in (("dot found", GREEN), ("2-inch gap", RED), ("maybe gap", ORANGE),
                         ("unreadable", MAGENTA), ("area checked", CYAN)):
        cv2.rectangle(legend, (x, 12), (x + 18, 30), color, -1)
        cv2.putText(legend, label, (x + 24, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        x += 40 + 11 * len(label)
    return np.vstack([banner, view, legend]), len(dots)
