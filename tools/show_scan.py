"""Draw one scan step on a photo: found dots, unreadable glass, gaps, and the decision.

Run: python -m tools.show_scan <photo> <out.png> [--dot-mm 19.05] [--canvas x y w h]
"""

import argparse

import cv2

from birdsafe.calib import Rejected, rectify
from birdsafe.markers import DOT_MM
from birdsafe.render import patch_canvas, render_step
from birdsafe.scan import ScanState, scan_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("out")
    parser.add_argument("--dot-mm", type=float, default=DOT_MM)
    parser.add_argument("--canvas", type=float, nargs=4, metavar=("X", "Y", "W", "H"))
    args = parser.parse_args()

    capture = rectify(cv2.imread(args.photo))
    if isinstance(capture, Rejected):
        print(f"rejected: {capture.reason}")
        return
    canvas = tuple(args.canvas) if args.canvas else patch_canvas(capture, args.dot_mm)
    if canvas is None:
        print("no dots found beside the sheet")
        return
    state = scan_step(ScanState(canvas_mm=canvas, dot_mm=args.dot_mm), capture)
    image, _dots = render_step(capture, state)
    cv2.imwrite(args.out, image)
    print(f"canvas {tuple(round(v, 1) for v in canvas)}: {state.trace[-1].decision.message}")


if __name__ == "__main__":
    main()
