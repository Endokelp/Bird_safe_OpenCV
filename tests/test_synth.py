from birdsafe.coverage import RULE_MM
from birdsafe.synth import grid_positions, make_pane


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
