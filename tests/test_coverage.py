import numpy as np
import pytest
from birdsafe.coverage import RULE_MM, check_coverage
from birdsafe.synth import make_pane


def _check(p, rule_mm=RULE_MM):
    return check_coverage(p.pane, p.markers, p.unknown, p.mm_per_px, rule_mm)


def test_tight_grid_passes():
    # 40 mm spacing with 6 mm dots leaves 34 mm of empty space, under the 50.8 mm threshold.
    r = _check(make_pane(300, 300, 40, 6, 0.5))
    assert not r.confirmed.any()
    assert r.targets == ()


def test_loose_grid_violates():
    # 70 mm spacing leaves 64 mm of empty space, over the threshold.
    r = _check(make_pane(300, 300, 70, 6, 0.5))
    assert r.confirmed.any()
    assert r.largest_gap_mm > RULE_MM


def test_a_single_missing_dot_in_a_tight_grid_is_found_and_located():
    # The omitted dot sits at 130 mm, so its neighbours leave a roughly 74 mm empty region.
    p = make_pane(300, 300, 40, 6, 0.5, omit=((3, 3),))
    r = _check(p)
    assert len(r.targets) == 1
    tx, ty = r.targets[0].center_mm
    gx, gy = p.planted_gap_centers_mm[0]
    assert abs(tx - gx) < 5.0 and abs(ty - gy) < 5.0


def test_unknown_region_demotes_a_violation_to_possible():
    # The unknown box must cover the omitted dot at 130 mm, so it spans 110 mm to 170 mm.
    p = make_pane(300, 300, 40, 6, 0.5, omit=((3, 3),),
                  unknown_boxes_mm=((110.0, 110.0, 60.0, 60.0),))
    r = _check(p)
    assert not r.confirmed.any()
    assert r.possible.any()


def test_unknown_fraction_is_reported():
    p = make_pane(300, 300, 40, 6, 0.5, unknown_boxes_mm=((0.0, 0.0, 150.0, 300.0),))
    assert _check(p).unknown_fraction == pytest.approx(0.5, abs=0.02)


def test_largest_gap_on_a_bare_pane_is_bounded_by_the_pane():
    # A spacing larger than the pane yields one centred dot; omitting it leaves the pane bare.
    p = make_pane(100, 100, 1000, 6, 0.5, omit=((0, 0),))
    r = _check(p)
    # The largest empty square in a bare 100 mm pane is the pane itself.
    assert r.largest_gap_mm == pytest.approx(100.0, abs=3.0)


@pytest.mark.parametrize("mm_per_px", [0.25, 0.4, 0.5, 1.0])
@pytest.mark.parametrize("spacing", [40, 45, 50, 55, 56, 57, 60])
def test_headline_number_never_contradicts_the_verdict(spacing, mm_per_px):
    # Spacings 50 to 57 mm straddle the threshold, where a circle-based number read over 2 inches
    # on panes the square test passed.
    r = _check(make_pane(400, 400, spacing, 6, mm_per_px))
    assert r.confirmed.any() == (r.largest_gap_mm > RULE_MM)


def test_unknown_area_is_never_counted_as_a_gap():
    # Detection finds no dots under glare, so the dots inside the box are absent from the mask.
    p = make_pane(400, 400, 40, 6, 0.5, unknown_boxes_mm=((100.0, 100.0, 150.0, 150.0),))
    markers = p.markers & ~p.unknown
    r = check_coverage(p.pane, markers, p.unknown, p.mm_per_px)
    assert r.largest_gap_mm < RULE_MM
    assert not r.confirmed.any()
    assert r.possible.any()


def test_opencv_style_uint8_masks_give_the_same_answer_as_booleans():
    p = make_pane(300, 300, 40, 6, 0.5, omit=((3, 3),),
                  unknown_boxes_mm=((0.0, 0.0, 150.0, 300.0),))
    as_u8 = [m.astype(np.uint8) * 255 for m in (p.pane, p.markers, p.unknown)]
    r = check_coverage(*as_u8, p.mm_per_px)
    expected = _check(p)
    assert r.unknown_fraction == pytest.approx(expected.unknown_fraction)
    assert r.largest_gap_mm == expected.largest_gap_mm
    assert r.confirmed.dtype == bool
    assert (r.confirmed == expected.confirmed).all() and (r.possible == expected.possible).all()


def test_a_marker_seen_under_glare_does_not_hide_a_possible_violation():
    p = make_pane(300, 300, 40, 6, 0.5, omit=((3, 3),),
                  unknown_boxes_mm=((110.0, 110.0, 60.0, 60.0),))
    markers = p.markers.copy()
    markers[258:262, 258:262] = True  # spurious detection at the planted gap centre, 130 mm
    r = check_coverage(p.pane, markers, p.unknown, p.mm_per_px)
    assert r.possible.any()


def test_scale_changes_the_answer_not_just_the_pixels():
    coarse = _check(make_pane(300, 300, 70, 6, 1.0))
    fine = _check(make_pane(300, 300, 70, 6, 0.25))
    assert abs(coarse.largest_gap_mm - fine.largest_gap_mm) < 4.0


def test_rejects_a_kernel_too_small_to_be_meaningful():
    with pytest.raises(ValueError):
        check_coverage(np.ones((10, 10), bool), np.zeros((10, 10), bool),
                       np.zeros((10, 10), bool), mm_per_px=100.0)
