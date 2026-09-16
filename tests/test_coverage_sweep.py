import pytest
from birdsafe.coverage import RULE_MM, check_coverage
from birdsafe.synth import make_pane


@pytest.mark.parametrize("mm_per_px", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("spacing", [30, 40, 45])
def test_no_false_alarm_on_compliant_panes(spacing, mm_per_px):
    p = make_pane(400, 400, spacing, 6, mm_per_px)
    assert not check_coverage(p.pane, p.markers, p.unknown, p.mm_per_px).confirmed.any()


@pytest.mark.parametrize("mm_per_px", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("spacing", [60, 70, 90])
def test_every_noncompliant_pane_is_caught(spacing, mm_per_px):
    p = make_pane(400, 400, spacing, 6, mm_per_px)
    r = check_coverage(p.pane, p.markers, p.unknown, p.mm_per_px)
    assert r.confirmed.any() and r.largest_gap_mm > RULE_MM


@pytest.mark.parametrize("mm_per_px", [0.25, 0.5])
def test_largest_gap_error_stays_within_one_millimetre(mm_per_px):
    # A bare 120 mm pane: the largest empty square is the pane, exactly 120 mm across.
    p = make_pane(120, 120, 1000, 6, mm_per_px, omit=((0, 0),))
    r = check_coverage(p.pane, p.markers, p.unknown, p.mm_per_px)
    assert abs(r.largest_gap_mm - 120.0) <= 1.0 + 2 * mm_per_px
