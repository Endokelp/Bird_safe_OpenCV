"""Reports spread across repeated captures of one pane. Phase 2 gate, part 2.

Run from the repository root as a module so birdsafe and tools import:
`python -m tools.repeatability captures/session01/*.jpg`
"""

import statistics
import sys

import cv2
import numpy as np

from birdsafe.calib import Rejected, rectify
from birdsafe.coverage import check_coverage
from birdsafe.markers import detect_markers
from birdsafe.quality import unknown_regions


def main(paths: list[str]) -> int:
    rms = []
    gaps = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"{path}: unreadable")
            continue
        rectified = rectify(image)
        if isinstance(rectified, Rejected):
            print(f"{path}: {rectified.reason}")
            continue
        rms.append(rectified.reproj_rms_mm)
        markers = detect_markers(rectified.image, rectified.mm_per_px)
        quality = unknown_regions(rectified.image, rectified.mm_per_px)
        # ponytail: no pane quadrilateral until Phase 6. Repeatability measures the
        # full rectified frame, which is enough to compare captures of one taped sheet.
        pane = np.ones(markers.mask.shape, bool)
        coverage = check_coverage(
            pane, markers.mask, quality.unknown | ~rectified.valid, rectified.mm_per_px
        )
        gaps.append(coverage.largest_gap_mm)
    if len(rms) < 2:
        print(f"only {len(rms)} usable captures, cannot report spread")
        return 1
    print(f"usable {len(rms)}/{len(paths)}")
    print(f"reprojection rms mm: mean {statistics.mean(rms):.3f} max {max(rms):.3f}")
    if len(gaps) >= 2:
        print(f"largest gap mm: mean {statistics.mean(gaps):.3f} max {max(gaps):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
