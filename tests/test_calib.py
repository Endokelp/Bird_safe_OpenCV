"""Tests for metric rectification. Builds a known-perspective scene image, no camera needed."""

import cv2
import numpy as np

from birdsafe.calib import rectify
from tools.charuco_sheet import SQUARE_MM, SQUARES_X, SQUARES_Y, birdsafe_board

MM_PER_PX = 0.5


def _warped_scene(mm_per_px=MM_PER_PX):
    board_w = round(SQUARES_X * SQUARE_MM / mm_per_px)
    board_h = round(SQUARES_Y * SQUARE_MM / mm_per_px)
    # Canvas stands in for the glass: the board covers only part of the capture, offset from
    # the corner, the way a taped sheet sits inside a pane.
    canvas_w, canvas_h = board_w * 2, board_h * 2
    scene = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    board_img = cv2.cvtColor(birdsafe_board().generateImage((board_w, board_h)), cv2.COLOR_GRAY2BGR)
    off_x, off_y = board_w // 4, board_h // 4
    scene[off_y : off_y + board_h, off_x : off_x + board_w] = board_img
    src = np.float32([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]])
    # Pull the top edge inward so the scene is viewed off-axis, as a phone capture would be.
    dst = np.float32([[60, 20], [canvas_w - 20, 0], [canvas_w, canvas_h], [0, canvas_h - 30]])
    return cv2.warpPerspective(scene, cv2.getPerspectiveTransform(src, dst), (canvas_w, canvas_h))


def test_rectify_recovers_the_scale_it_was_asked_for():
    r = rectify(_warped_scene(), mm_per_px=MM_PER_PX)
    assert r is not None
    assert r.mm_per_px == MM_PER_PX


def test_rectify_finds_corners_and_reports_low_reprojection_error():
    r = rectify(_warped_scene())
    assert r is not None
    assert r.corners_found >= 6
    assert r.reproj_rms_px < 2.0


def test_rectified_board_squares_measure_the_physical_square_size():
    r = rectify(_warped_scene(), mm_per_px=MM_PER_PX)
    assert r is not None

    detector = cv2.aruco.CharucoDetector(birdsafe_board())
    corners, ids, _marker_corners, _marker_ids = detector.detectBoard(r.image)
    assert corners is not None and ids is not None

    by_id = dict(zip(ids.flatten().tolist(), corners.reshape(-1, 2)))
    row_width = SQUARES_X - 1  # inner corners per row
    distances_mm = []
    for cid, pt in by_id.items():
        if cid % row_width == row_width - 1:
            continue  # last column in its row, no right-hand neighbour to pair with
        neighbour = by_id.get(cid + 1)
        if neighbour is None:
            continue
        distances_mm.append(float(np.linalg.norm(pt - neighbour)) * r.mm_per_px)

    assert distances_mm
    mean_mm = sum(distances_mm) / len(distances_mm)
    assert abs(mean_mm - SQUARE_MM) < 0.5, f"observed mean square size {mean_mm:.3f} mm"


def test_rectified_view_covers_the_scene_beyond_the_board():
    r = rectify(_warped_scene(), mm_per_px=MM_PER_PX)
    assert r is not None
    board_w_px = SQUARES_X * SQUARE_MM / MM_PER_PX
    board_h_px = SQUARES_Y * SQUARE_MM / MM_PER_PX
    assert r.image.shape[1] > board_w_px
    assert r.image.shape[0] > board_h_px


def test_rectify_returns_none_when_no_board_is_present():
    assert rectify(np.zeros((400, 400, 3), np.uint8)) is None
