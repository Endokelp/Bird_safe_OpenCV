import cv2
import numpy as np

from birdsafe.coverage import RULE_MM
from birdsafe.synth import grid_positions, make_pane, render_pane


def test_full_grid_has_no_planted_gaps():
    p = make_pane(300, 300, 40, 6, 0.5)
    assert p.planted_gap_centers_mm == ()
    assert p.markers.any()
    assert p.pane.all()


def test_grid_tiles_the_pane_at_the_nominal_spacing():
    # 300 mm at 40 mm needs 8 columns, leaving a 10 mm margin on each side.
    assert grid_positions(300, 40) == [10.0, 50.0, 90.0, 130.0, 170.0, 210.0, 250.0, 290.0]


def test_edge_margin_never_reaches_the_rule_threshold():
    # A margin wider than the rule would make a compliant fixture fail for the wrong reason.
    for extent in (200, 300, 400, 517):
        for spacing in (30, 40, 45):
            assert grid_positions(extent, spacing)[0] < RULE_MM


def test_omitting_a_dot_records_its_center():
    p = make_pane(300, 300, 40, 6, 0.5, omit=((2, 2),))
    assert len(p.planted_gap_centers_mm) == 1
    x, y = p.planted_gap_centers_mm[0]
    assert abs(x - 90.0) < 1e-6 and abs(y - 90.0) < 1e-6


def test_omitted_dot_is_actually_absent_from_the_mask():
    full = make_pane(300, 300, 40, 6, 0.5)
    holed = make_pane(300, 300, 40, 6, 0.5, omit=((2, 2),))
    assert holed.markers.sum() < full.markers.sum()


def test_unknown_box_lands_in_millimetre_coordinates():
    p = make_pane(300, 300, 40, 6, 0.5, unknown_boxes_mm=((0, 0, 50, 50),))
    assert p.unknown[0, 0]
    assert not p.unknown[-1, -1]
    assert abs(p.unknown.sum() * 0.5 * 0.5 - 2500.0) < 50.0


def test_render_matches_the_marker_mask_shape_and_is_bgr():
    p = make_pane(300, 300, 40, 6, 0.5)
    image = render_pane(p)
    assert image.shape == (p.markers.shape[0], p.markers.shape[1], 3)
    assert image.dtype == np.uint8


def test_dark_dots_are_darker_than_the_background():
    p = make_pane(300, 300, 40, 6, 0.5)
    image = render_pane(p, dark_on_light=True, texture_std=0.0)
    gray = image.mean(axis=2)
    assert gray[p.markers].mean() < gray[~p.markers].mean()


def test_light_dots_are_brighter_than_the_background():
    p = make_pane(300, 300, 40, 6, 0.5)
    image = render_pane(p, dark_on_light=False, texture_std=0.0)
    gray = image.mean(axis=2)
    assert gray[p.markers].mean() > gray[~p.markers].mean()


def test_glare_box_is_saturated_and_hides_dots():
    p = make_pane(300, 300, 40, 6, 0.5)
    image = render_pane(p, texture_std=0.0, glare_boxes_mm=((20.0, 20.0, 100.0, 100.0),))
    x0, y0 = round(20.0 / p.mm_per_px), round(20.0 / p.mm_per_px)
    x1, y1 = round(120.0 / p.mm_per_px), round(120.0 / p.mm_per_px)
    patch = image[y0 + 10:y1 - 10, x0 + 10:x1 - 10]
    assert patch.min() == 255
    assert patch.max() == 255


def test_blur_lowers_high_frequency_energy():
    p = make_pane(300, 300, 40, 6, 0.5)
    sharp = render_pane(p, texture_std=8.0, blur_sigma_px=0.0)
    blurry = render_pane(p, texture_std=8.0, blur_sigma_px=8.0)
    # Laplacian energy is the same measure quality.py uses to decide a region cannot resolve a dot.
    assert _laplacian_energy(blurry) < 0.01 * _laplacian_energy(sharp)


def _laplacian_energy(image):
    gray = image.mean(axis=2)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float((lap * lap).mean())
