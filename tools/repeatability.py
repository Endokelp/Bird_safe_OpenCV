"""Reports spread across repeated captures of one pane. Phase 2 gate, part 2.

Run from the repository root as a module so birdsafe and tools import:
`python -m tools.repeatability captures/session01/*.jpg`
"""

import statistics
import sys

import cv2

from birdsafe.calib import rectify


def main(paths: list[str]) -> int:
    rms = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"{path}: unreadable")
            continue
        r = rectify(image)
        if r is None:
            print(f"{path}: no usable board view (not found, too oblique, or partly hidden)")
            continue
        rms.append(r.reproj_rms_mm)
    if len(rms) < 2:
        print(f"only {len(rms)} usable captures, cannot report spread")
        return 1
    print(f"usable {len(rms)}/{len(paths)}")
    print(f"reprojection rms mm: mean {statistics.mean(rms):.3f} max {max(rms):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
