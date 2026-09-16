"""Tests for metric rectification. Builds a known-perspective scene image, no camera needed."""

import cv2
import numpy as np
import pytest

from birdsafe.calib import CORNERS_PER_ROW, rectify
from tools.charuco_sheet import SQUARE_MM, SQUARES_X, SQUARES_Y, birdsafe_board

MM_PER_PX = 0.5
BOARD_W = round(SQUARES_X * SQUARE_MM / MM_PER_PX)
BOARD_H = round(SQUARES_Y * SQUARE_MM / MM_PER_PX)
# A red dot on the glass beyond the board's bottom-right corner, 230 mm right and 170 mm down from
# the board's top-left corner.
DOT_FROM_BOARD_MM = (230.0, 170.0)

VIEW_A = [[60, 20], [-20, 0], [0, 0], [0, -30]]
VIEW_B = [[0, 40], [-50, 10], [-10, 0], [30, -10]]


def _warped_scene(view=VIEW_A, square_rows_visible=SQUARES_Y):
    # Canvas stands in for the glass: the board covers only part of the capture, offset from
    # the corner, the way a taped sheet sits inside a pane.
    canvas_w, canvas_h = BOARD_W * 2, BOARD_H * 2
    scene = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    board_img = cv2.cvtColor(birdsafe_board().generateImage((BOARD_W, BOARD_H)), cv2.COLOR_GRAY2BGR)
    board_img[round(square_rows_visible * SQUARE_MM / MM_PER_PX):] = 255
    off_x, off_y = BOARD_W // 4, BOARD_H // 4
    scene[off_y : off_y + BOARD_H, off_x : off_x + BOARD_W] = board_img
    dot = (off_x + round(DOT_FROM_BOARD_MM[0] / MM_PER_PX), off_y + round(DOT_FROM_BOARD_MM[1] / MM_PER_PX))
    cv2.circle(scene, dot, 8, (0, 0, 255), -1)

    src = np.float32([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]])
    # Offsets are relative to each canvas corner, so every view looks at the pane off-axis.
    dst = src + np.float32(view)
    return cv2.warpPerspective(scene, cv2.getPerspectiveTransform(src, dst), (canvas_w, canvas_h))


def _red_dot_mm(r):
    b, g, red = cv2.split(r.image)
    ys, xs = np.nonzero((red > 180) & (g < 80) & (b < 80))
    assert len(xs) > 0
    return xs.mean() * r.mm_per_px, ys.mean() * r.mm_per_px


def test_rectify_finds_corners_and_reports_low_reprojection_error():
    r = rectify(_warped_scene())
    assert r is not None
    assert r.corners_found >= 6
    assert r.reproj_rms_mm < 1.0


def test_rectified_board_squares_measure_the_physical_square_size():
    r = rectify(_warped_scene(), mm_per_px=MM_PER_PX)
    assert r is not None

    corners, ids, _marker_corners, _marker_ids = cv2.aruco.CharucoDetector(
        birdsafe_board()
    ).detectBoard(r.image)
    assert corners is not None and ids is not None

    by_id = dict(zip(ids.flatten().tolist(), corners.reshape(-1, 2)))
    distances_mm = [
        float(np.linalg.norm(pt - by_id[cid + 1])) * r.mm_per_px
        for cid, pt in by_id.items()
        if cid % CORNERS_PER_ROW != CORNERS_PER_ROW - 1 and cid + 1 in by_id
    ]
    assert distances_mm
    mean_mm = sum(distances_mm) / len(distances_mm)
    assert abs(mean_mm - SQUARE_MM) < 0.5, f"observed mean square size {mean_mm:.3f} mm"


@pytest.mark.parametrize("view", [VIEW_A, VIEW_B])
def test_a_point_on_the_glass_lands_at_the_same_board_relative_position_in_every_view(view):
    r = rectify(_warped_scene(view))
    assert r is not None
    x_mm, y_mm = _red_dot_mm(r)
    origin_x, origin_y = r.board_origin_mm
    assert abs((x_mm - origin_x) - DOT_FROM_BOARD_MM[0]) < 1.0
    assert abs((y_mm - origin_y) - DOT_FROM_BOARD_MM[1]) < 1.0


def test_rectified_view_covers_the_scene_beyond_the_board():
    r = rectify(_warped_scene(), mm_per_px=MM_PER_PX)
    assert r is not None
    assert r.image.shape[1] > BOARD_W
    assert r.image.shape[0] > BOARD_H


def test_rectify_rejects_corners_from_a_single_board_row():
    # Glare hides all but the top two square rows, leaving one collinear row of inner corners.
    assert rectify(_warped_scene(square_rows_visible=2)) is None


def test_rectify_still_measures_when_two_corner_rows_survive():
    assert rectify(_warped_scene(square_rows_visible=3)) is not None


def test_rectify_rejects_an_output_too_large_to_be_a_sane_view():
    assert rectify(_warped_scene(), mm_per_px=0.05) is None


def test_rectify_returns_none_when_no_board_is_present():
    assert rectify(np.zeros((400, 400, 3), np.uint8)) is None
