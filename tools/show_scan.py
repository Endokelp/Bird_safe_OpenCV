"""Draw one scan step on a photo: found dots, unreadable glass, gaps, and the decision.

Run: python -m tools.show_scan <photo> <out.png> [--dot-mm 19.05] [--canvas x y w h]
"""

import argparse

import cv2
import numpy as np

from birdsafe.calib import Rejected, rectify
from birdsafe.coverage import check_coverage
from birdsafe.fusion import fuse
from birdsafe.markers import DOT_MM, detect_markers
from birdsafe.scan import ScanState, _region, scan_step

GREEN, MAGENTA, RED, ORANGE, CYAN = (0, 220, 0), (255, 0, 255), (0, 0, 255), (0, 165, 255), (255, 255, 0)
BANNER = {"pass": (40, 150, 40), "repair": (30, 30, 200), "recapture": (0, 140, 230)}
# ponytail: patch canvas = found dots +/- this margin; pass --canvas once real panes have edges.
MARGIN_MM = 12


def _tint(image, mask, color, alpha=0.5):
    out = image.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def _canvas_from_dots(capture, dot_mm):
    ox, oy = capture.board_origin_mm
    centers = [(x - ox, y - oy) for x, y in detect_markers(capture.image, capture.mm_per_px, dot_mm).centers_mm]
    # The sheet's own squares pass the dot filter; only dots right of the sheet are the patch.
    xs, ys = zip(*[c for c in centers if c[0] > 215], strict=True)
    return (min(xs) - MARGIN_MM, min(ys) - MARGIN_MM,
            max(xs) - min(xs) + 2 * MARGIN_MM, max(ys) - min(ys) + 2 * MARGIN_MM)


def render(photo, dot_mm, canvas_mm=None):
    capture = rectify(photo)
    if isinstance(capture, Rejected):
        return None, f"rejected: {capture.reason}"
    canvas_mm = canvas_mm or _canvas_from_dots(capture, dot_mm)
    state = scan_step(ScanState(canvas_mm=canvas_mm, dot_mm=dot_mm), capture)
    record = state.trace[-1]
    scale = state.mm_per_px

    fused = fuse(state.observations, canvas_mm, mm_per_px=scale)
    region = _region(state, fused.unknown.shape)
    coverage = check_coverage(region, fused.markers, fused.unknown, scale)

    # Canvas pixel (0, 0) sits here in the rectified image, the same shift fuse() applies.
    sx = round((canvas_mm[0] + capture.board_origin_mm[0]) / scale)
    sy = round((canvas_mm[1] + capture.board_origin_mm[1]) / scale)
    h, w = fused.unknown.shape
    full = np.zeros(capture.image.shape[:2], bool)

    def place(mask):
        out = full.copy()
        out[sy:sy + h, sx:sx + w] = mask[: out.shape[0] - sy, : out.shape[1] - sx]
        return out

    view = _tint(capture.image, place(fused.unknown & region), MAGENTA)
    view = _tint(view, place(coverage.possible), ORANGE)
    view = _tint(view, place(coverage.confirmed), RED, 0.55)
    dots, _ = cv2.findContours(place(fused.markers).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, dots, -1, GREEN, 3)
    cv2.rectangle(view, (sx, sy), (sx + w, sy + h), CYAN, 2)
    for x_mm, y_mm, _area in record.metrics.repair_targets:
        cx, cy = round(sx + x_mm / scale), round(sy + y_mm / scale)
        cv2.drawMarker(view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 40, 3)

    m, d = record.metrics, record.decision
    lines = [
        f"{d.action.upper()}  ({d.rule})   largest gap {m.largest_gap_mm:.0f} mm (limit 50.8)",
        f"dots found {len(dots)}   unreadable {m.unknown_fraction:.0%}   board fit {m.reproj_rms_mm:.2f} mm"
        f"   view {m.view_angle_deg:.0f} deg",
    ]
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
    return np.vstack([banner, view, legend]), d.message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("out")
    parser.add_argument("--dot-mm", type=float, default=DOT_MM)
    parser.add_argument("--canvas", type=float, nargs=4, metavar=("X", "Y", "W", "H"))
    args = parser.parse_args()
    image, message = render(cv2.imread(args.photo), args.dot_mm, tuple(args.canvas) if args.canvas else None)
    if image is not None:
        cv2.imwrite(args.out, image)
    print(message)


if __name__ == "__main__":
    main()
