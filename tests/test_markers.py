import math

import cv2

from birdsafe.markers import DOT_MM, detect_markers
from birdsafe.synth import grid_positions, make_pane, render_pane


def _truth_centers_mm(width_mm, height_mm, spacing_mm, omit=()):
    omitted = set(omit)
    xs = grid_positions(width_mm, spacing_mm)
    ys = grid_positions(height_mm, spacing_mm)
    return [
        (x_mm, y_mm)
        for row, y_mm in enumerate(ys)
        for col, x_mm in enumerate(xs)
        if (row, col) not in omitted
    ]


def _match(detected, truth, tol_mm=DOT_MM):
    unused = list(detected)
    hits = 0
    for tx, ty in truth:
        best_i = None
        best_d = tol_mm
        for i, (dx, dy) in enumerate(unused):
            dist = math.hypot(dx - tx, dy - ty)
            if dist <= best_d:
                best_d = dist
                best_i = i
        if best_i is not None:
            hits += 1
            unused.pop(best_i)
    return hits, len(unused), len(truth) - hits


def _recall_case(dark_on_light, mm_per_px=0.5, omit=()):
    pane = make_pane(300, 300, 40, DOT_MM, mm_per_px, omit=omit)
    image = render_pane(pane, dark_on_light=dark_on_light)
    result = detect_markers(image, pane.mm_per_px)
    truth = _truth_centers_mm(300, 300, 40, omit)
    return _match(result.centers_mm, truth) + (result,)


def test_dark_dots_on_light_are_all_found():
    tp, fp, fn, result = _recall_case(True)
    assert fn == 0 and fp == 0 and tp == 64
    assert result.mask.shape[:2] == (600, 600)


def test_light_dots_on_dark_are_all_found():
    tp, fp, fn, _result = _recall_case(False)
    assert fn == 0 and fp == 0 and tp == 64


def test_an_omitted_dot_is_not_detected():
    tp, fp, fn, _result = _recall_case(True, omit=((2, 2),))
    assert fn == 0 and fp == 0 and tp == 63


def test_detection_survives_downscale_to_working_resolution():
    tp, fp, fn, _result = _recall_case(True, mm_per_px=0.25)
    assert fn == 0 and fp == 0 and tp == 64


def test_a_large_square_is_not_a_dot():
    pane = make_pane(300, 300, 40, DOT_MM, 0.5)
    image = render_pane(pane, texture_std=0.0)
    x0, y0 = round(20 / pane.mm_per_px), round(20 / pane.mm_per_px)
    side = round(40 / pane.mm_per_px)
    cv2.rectangle(image, (x0, y0), (x0 + side, y0 + side), (20, 20, 20), -1)
    result = detect_markers(image, pane.mm_per_px)
    square_center = (20 + 20, 20 + 20)
    near = [
        (x_mm, y_mm)
        for x_mm, y_mm in result.centers_mm
        if math.hypot(x_mm - square_center[0], y_mm - square_center[1]) < 15
    ]
    assert near == []


def test_large_dots_on_a_dense_grid_stay_separate():
    # Session01 geometry: 3/4 inch dots at 40 mm. The glass between them is narrower than the
    # hat kernel, so a merged polarity mask turned the whole grid into one blob.
    dot_mm = 19.05
    pane = make_pane(300, 300, 40, dot_mm, 0.5)
    result = detect_markers(render_pane(pane, dark_on_light=False), pane.mm_per_px, dot_mm)
    tp, fp, fn = _match(result.centers_mm, _truth_centers_mm(300, 300, 40))
    assert fn == 0 and fp == 0 and tp == 64
