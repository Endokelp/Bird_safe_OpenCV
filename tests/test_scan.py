import numpy as np
import pytest
from birdsafe.calib import Rectified, Rejected
from birdsafe.coverage import check_coverage
from birdsafe.markers import DOT_MM
from birdsafe.policy import replay, trace_from_jsonl, trace_to_jsonl
from birdsafe.scan import ScanState, scan_step
from birdsafe.synth import grid_positions, make_pane, render_pane
from tools.charuco_sheet import sheet_footprint_mm

MM_PER_PX = 0.5
# The canvas sits right of the calibration sheet, so the sheet never overlaps it.
CANVAS_CLEAR_OF_SHEET_MM = (300.0, 0.0, 300.0, 300.0)
ORIGIN_CLEAR_OF_SHEET_MM = (-300.0, 0.0)

PANE = make_pane(300, 300, 40, DOT_MM, MM_PER_PX, omit=((3, 3),))
REPAIRED_PANE = make_pane(300, 300, 40, DOT_MM, MM_PER_PX)
GAP_X_MM, GAP_Y_MM = PANE.planted_gap_centers_mm[0]
# 100 mm boxes keep the unknown fraction well above MAX_UNKNOWN_FRACTION; the glare mask loses
# about half a quality window on each side, and an 80 mm box sat at 0.052 against 0.05.
GLARE_ON_GAP = ((GAP_X_MM - 50.0, GAP_Y_MM - 50.0, 100.0, 100.0),)
GLARE_AWAY_FROM_GAP = ((195.0, 195.0, 100.0, 100.0),)


def _capture(pane, glare=(), origin_mm=ORIGIN_CLEAR_OF_SHEET_MM, view_angle_deg=0.0):
    image = render_pane(pane, glare_boxes_mm=glare)
    return Rectified(
        image=image,
        mm_per_px=MM_PER_PX,
        board_origin_mm=origin_mm,
        reproj_rms_mm=0.06,
        corners_found=24,
        valid=np.ones(image.shape[:2], bool),
        native_mm_per_px=MM_PER_PX,
        view_angle_deg=view_angle_deg,
    )


def _new_state():
    return ScanState(canvas_mm=CANVAS_CLEAR_OF_SHEET_MM)


@pytest.fixture(scope="module")
def loop_state():
    state = _new_state()
    for pane, glare in ((PANE, GLARE_ON_GAP), (PANE, GLARE_AWAY_FROM_GAP), (REPAIRED_PANE, ())):
        state = scan_step(state, _capture(pane, glare))
    return state


def test_glare_then_a_clear_view_then_the_repair_walks_recapture_repair_pass(loop_state):
    glared, resolved, repaired = loop_state.trace

    assert (glared.decision.action, glared.decision.rule) == ("recapture", "unknown_area")
    aim_x, aim_y = glared.decision.aim_mm
    assert abs(aim_x - GAP_X_MM) < 20.0 and abs(aim_y - GAP_Y_MM) < 20.0

    assert (resolved.decision.action, resolved.decision.rule) == ("repair", "confirmed_gap")
    assert resolved.metrics.unknown_fraction < glared.metrics.unknown_fraction
    assert {name: (b, a) for name, b, a in resolved.effect}["unknown_fraction"] == (
        glared.metrics.unknown_fraction,
        resolved.metrics.unknown_fraction,
    )
    (target,) = resolved.decision.targets_mm
    assert abs(target[0] - GAP_X_MM) < 10.0 and abs(target[1] - GAP_Y_MM) < 10.0

    assert (repaired.decision.action, repaired.decision.rule) == ("pass", "all_clear")
    assert repaired.metrics.frames == 1
    assert repaired.decision.fixed_mm == ((target[0], target[1]),)
    assert repaired.decision.message.startswith("The repaired spot now passes.")


def test_the_trace_round_trips_through_jsonl_and_replays_to_its_decisions(loop_state):
    trace = list(loop_state.trace)
    assert trace_from_jsonl(trace_to_jsonl(trace)) == trace
    assert replay(trace) == [r.decision for r in trace]


def test_an_oblique_capture_is_not_fused():
    state = scan_step(_new_state(), _capture(PANE, GLARE_ON_GAP))
    after = scan_step(state, _capture(PANE, view_angle_deg=60.0))

    assert after.trace[-1].decision.rule == "too_oblique"
    assert len(after.observations) == len(state.observations) == 1
    assert after.trace[-1].metrics.frames == 1


def test_a_rejected_capture_is_a_recapture_with_the_reason_and_an_unread_pane():
    state = scan_step(_new_state(), Rejected("board_not_found"))

    decision = state.trace[0].decision
    assert (decision.action, decision.rule) == ("recapture", "board_not_found")
    assert state.observations == ()
    assert state.trace[0].metrics.frames == 0
    assert state.trace[0].metrics.unknown_fraction == 1.0


def test_the_same_glare_four_times_stops_at_the_loop_cap():
    capture = _capture(PANE, GLARE_ON_GAP)
    state = _new_state()
    for _ in range(4):
        state = scan_step(state, capture)

    assert [r.decision.action for r in state.trace] == ["recapture"] * 3 + ["stop"]
    assert state.trace[-1].decision.rule == "loop_cap"


def test_an_unmarked_sheet_area_is_not_a_repair_target():
    # Dots are left off exactly where the sheet lies, so the sheet area is one large gap.
    x0, y0, x1, y1 = sheet_footprint_mm()
    positions = grid_positions(300.0, 40.0)
    under_sheet = [
        (row, col)
        for row, y in enumerate(positions)
        for col, x in enumerate(positions)
        if x <= x1 and y <= y1
    ]
    pane = make_pane(300, 300, 40, DOT_MM, MM_PER_PX, omit=under_sheet)
    unchecked = check_coverage(pane.pane, pane.markers, pane.unknown, MM_PER_PX)
    assert any(x0 <= t.center_mm[0] <= x1 and y0 <= t.center_mm[1] <= y1 for t in unchecked.targets)

    state = scan_step(
        ScanState(canvas_mm=(0.0, 0.0, 300.0, 300.0)), _capture(pane, origin_mm=(0.0, 0.0))
    )

    assert state.trace[0].metrics.repair_targets == ()
    assert state.trace[0].decision.action == "pass"
