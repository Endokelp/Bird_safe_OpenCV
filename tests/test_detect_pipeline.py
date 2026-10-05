import cv2

from birdsafe.coverage import RULE_MM, check_coverage
from birdsafe.markers import DOT_MM, detect_markers
from birdsafe.quality import unknown_regions
from birdsafe.synth import make_pane, render_pane
from tools.repeatability import main as repeatability_main


def test_a_missing_dot_on_a_rendered_pane_is_a_confirmed_repair():
    pane = make_pane(300, 300, 40, DOT_MM, 0.5, omit=((3, 3),))
    image = render_pane(pane)
    markers = detect_markers(image, pane.mm_per_px)
    quality = unknown_regions(image, pane.mm_per_px)
    result = check_coverage(pane.pane, markers.mask, quality.unknown, pane.mm_per_px)
    assert result.confirmed.any()
    assert result.largest_gap_mm > RULE_MM
    assert len(result.targets) == 1
    gx, gy = pane.planted_gap_centers_mm[0]
    tx, ty = result.targets[0].center_mm
    assert abs(tx - gx) < 10.0 and abs(ty - gy) < 10.0


def test_glare_over_a_missing_dot_demotes_the_finding_to_possible():
    pane = make_pane(300, 300, 40, DOT_MM, 0.5, omit=((3, 3),))
    gx, gy = pane.planted_gap_centers_mm[0]
    image = render_pane(
        pane,
        glare_boxes_mm=((gx - 40.0, gy - 40.0, 80.0, 80.0),),
    )
    markers = detect_markers(image, pane.mm_per_px)
    quality = unknown_regions(image, pane.mm_per_px)
    result = check_coverage(pane.pane, markers.mask, quality.unknown, pane.mm_per_px)
    assert not result.confirmed.any()
    assert result.possible.any()


def test_repeatability_reports_gap_spread_on_two_usable_views(tmp_path, capsys):
    from birdsafe.calib import Rectified, rectify
    from tests.test_calib import VIEW_A, VIEW_B, _warped_scene

    paths = []
    for i, view in enumerate((VIEW_A, VIEW_B)):
        path = tmp_path / f"view{i}.png"
        cv2.imwrite(str(path), _warped_scene(view))
        paths.append(str(path))
        assert isinstance(rectify(_warped_scene(view)), Rectified)

    assert repeatability_main(paths) == 0
    out = capsys.readouterr().out
    assert "reprojection rms mm:" in out
    assert "largest gap mm:" in out
