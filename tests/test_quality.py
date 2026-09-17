from birdsafe.markers import DOT_MM
from birdsafe.quality import unknown_regions
from birdsafe.synth import make_pane, render_pane


def test_sharp_textured_pane_is_almost_fully_observed():
    pane = make_pane(300, 300, 40, DOT_MM, 0.5)
    image = render_pane(pane, texture_std=8.0, blur_sigma_px=0.0)
    result = unknown_regions(image, pane.mm_per_px)
    assert result.unknown.mean() < 0.02
    assert result.glare.mean() < 0.02
    assert result.blur.mean() < 0.02


def test_planted_glare_is_marked_unknown_and_is_glare():
    pane = make_pane(300, 300, 40, DOT_MM, 0.5)
    box = (20.0, 20.0, 100.0, 100.0)
    image = render_pane(pane, texture_std=8.0, glare_boxes_mm=(box,))
    result = unknown_regions(image, pane.mm_per_px)
    x = round((20.0 + 50.0) / pane.mm_per_px)
    y = round((20.0 + 50.0) / pane.mm_per_px)
    assert result.glare[y, x]
    assert result.unknown[y, x]
    far_x = round(250.0 / pane.mm_per_px)
    far_y = round(250.0 / pane.mm_per_px)
    assert not result.glare[far_y, far_x]


def test_heavy_blur_marks_the_pane_unknown():
    pane = make_pane(300, 300, 40, DOT_MM, 0.5)
    image = render_pane(pane, texture_std=8.0, blur_sigma_px=8.0)
    result = unknown_regions(image, pane.mm_per_px)
    assert result.blur.mean() > 0.9
    assert result.unknown.mean() > 0.9
    assert result.glare.mean() < 0.05
