import pytest
from birdsafe.coverage import RULE_MM
from birdsafe.markers import DOT_MM
from birdsafe.policy import (
    MAX_NATIVE_MM_PER_PX,
    MAX_REPROJ_RMS_MM,
    MAX_STALLED_RECAPTURES,
    MAX_UNKNOWN_FRACTION,
    MAX_VIEW_ANGLE_DEG,
    CaptureMetrics,
    capture_rule,
    decide,
    record,
    replay,
    trace_from_jsonl,
    trace_to_jsonl,
)

TARGET = (130.0, 130.0, 1500.0)
OTHER_TARGET = (230.0, 230.0, 900.0)
REJECT_REASONS = ["board_not_found", "board_partly_hidden", "view_too_oblique", "view_too_large"]


def _metrics(**overrides):
    fields = dict(
        reject_reason=None,
        reproj_rms_mm=0.1,
        view_angle_deg=10.0,
        native_mm_per_px=0.5,
        frames=1,
        unknown_fraction=0.0,
        unknown_centroid_mm=None,
        possible_centroids_mm=(),
        repair_targets=(),
        largest_gap_mm=30.0,
    )
    return CaptureMetrics(**{**fields, **overrides})


def _rejected(reason):
    return _metrics(
        reject_reason=reason,
        reproj_rms_mm=None,
        view_angle_deg=None,
        native_mm_per_px=None,
        frames=0,
        unknown_fraction=1.0,
        largest_gap_mm=0.0,
    )


def _glare(**overrides):
    return _metrics(unknown_fraction=0.2, unknown_centroid_mm=(150.0, 80.0), **overrides)


def _trace(*captures):
    history = []
    for metrics in captures:
        history.append(record(metrics, history))
    return history


def _actions(trace):
    return [r.decision.action for r in trace]


def test_a_clean_capture_has_no_capture_rule():
    assert capture_rule(_metrics()) is None


@pytest.mark.parametrize("reason", REJECT_REASONS)
def test_a_rejected_capture_returns_its_reject_reason(reason):
    assert capture_rule(_rejected(reason)) == reason


def test_a_bent_board_is_board_not_flat():
    assert capture_rule(_metrics(reproj_rms_mm=MAX_REPROJ_RMS_MM + 0.1)) == "board_not_flat"


def test_a_steep_view_is_too_oblique():
    assert capture_rule(_metrics(view_angle_deg=MAX_VIEW_ANGLE_DEG + 1)) == "too_oblique"


def test_a_coarse_capture_is_too_far():
    assert capture_rule(_metrics(native_mm_per_px=MAX_NATIVE_MM_PER_PX + 0.1)) == "too_far"


def test_the_coarse_limit_is_four_pixels_across_a_dot():
    assert MAX_NATIVE_MM_PER_PX * 4 == DOT_MM


def test_capture_rules_fire_in_a_fixed_order():
    everything = _metrics(
        reproj_rms_mm=MAX_REPROJ_RMS_MM + 1,
        view_angle_deg=MAX_VIEW_ANGLE_DEG + 1,
        native_mm_per_px=MAX_NATIVE_MM_PER_PX + 1,
    )
    assert capture_rule(everything) == "board_not_flat"
    assert capture_rule(_metrics(view_angle_deg=90.0, native_mm_per_px=9.0)) == "too_oblique"
    rejected_and_bent = _metrics(reject_reason="board_not_found", reproj_rms_mm=9.0)
    assert capture_rule(rejected_and_bent) == "board_not_found"


@pytest.mark.parametrize("reason", REJECT_REASONS)
def test_a_rejected_capture_recaptures_with_no_aim_point(reason):
    d = decide(_rejected(reason), [])
    assert (d.action, d.rule, d.aim_mm, d.targets_mm) == ("recapture", reason, None, ())


@pytest.mark.parametrize(
    ("overrides", "rule"),
    [
        ({"reproj_rms_mm": MAX_REPROJ_RMS_MM + 1}, "board_not_flat"),
        ({"view_angle_deg": MAX_VIEW_ANGLE_DEG + 1}, "too_oblique"),
        ({"native_mm_per_px": MAX_NATIVE_MM_PER_PX + 1}, "too_far"),
    ],
)
def test_a_capture_check_failure_recaptures_under_its_rule(overrides, rule):
    d = decide(_metrics(**overrides), [])
    assert (d.action, d.rule, d.aim_mm) == ("recapture", rule, None)


def test_unknown_area_recaptures_aimed_at_the_unknown_centroid():
    d = decide(_glare(), [])
    assert (d.action, d.rule, d.aim_mm) == ("recapture", "unknown_area", (150.0, 80.0))


def test_unknown_at_the_threshold_is_not_unknown_area():
    d = decide(_metrics(unknown_fraction=MAX_UNKNOWN_FRACTION), [])
    assert d.rule == "all_clear"


def test_a_possible_gap_recaptures_aimed_at_the_first_possible_centroid():
    d = decide(_metrics(possible_centroids_mm=((120.0, 140.0), (30.0, 30.0))), [])
    assert (d.action, d.rule, d.aim_mm) == ("recapture", "possible_gap", (120.0, 140.0))


def test_a_confirmed_gap_requests_repair_at_the_targets():
    d = decide(_metrics(repair_targets=(TARGET, OTHER_TARGET), largest_gap_mm=74.0), [])
    assert (d.action, d.rule) == ("repair", "confirmed_gap")
    assert d.targets_mm == (TARGET, OTHER_TARGET)
    assert d.aim_mm is None


def test_a_confirmed_gap_beats_a_possible_gap():
    d = decide(_metrics(repair_targets=(TARGET,), possible_centroids_mm=((10.0, 10.0),)), [])
    assert d.rule == "confirmed_gap"


def test_no_findings_passes():
    d = decide(_metrics(), [])
    assert (d.action, d.rule) == ("pass", "all_clear")
    assert (d.aim_mm, d.targets_mm, d.fixed_mm) == (None, (), ())


def test_a_capture_failure_beats_a_confirmed_gap():
    d = decide(_metrics(view_angle_deg=60.0, repair_targets=(TARGET,)), [])
    assert (d.action, d.rule, d.targets_mm) == ("recapture", "too_oblique", ())


def test_unknown_area_beats_a_confirmed_gap():
    d = decide(_glare(repair_targets=(TARGET,)), [])
    assert d.rule == "unknown_area"


def test_loop_cap_fires_on_the_third_stalled_recapture_and_not_before():
    trace = _trace(*[_glare()] * (MAX_STALLED_RECAPTURES + 1))
    assert _actions(trace) == ["recapture"] * MAX_STALLED_RECAPTURES + ["stop"]
    assert trace[-1].decision.rule == "loop_cap"


def test_loop_cap_reports_the_targets_it_has():
    trace = _trace(*[_glare(repair_targets=(TARGET,))] * 4)
    assert trace[-1].decision.targets_mm == (TARGET,)


def test_an_improving_recapture_is_not_counted_as_stalled():
    first = _metrics(unknown_fraction=0.4, unknown_centroid_mm=(150.0, 80.0))
    trace = _trace(first, *[_glare()] * 4)
    assert _actions(trace) == ["recapture"] * 4 + ["stop"]


def test_a_recapture_that_cleared_a_capture_failure_is_not_counted_as_stalled():
    rejected = _rejected("board_not_found")
    trace = _trace(rejected, rejected, rejected, _metrics(unknown_fraction=1.0))
    assert _actions(trace) == ["recapture"] * 4


def test_the_cap_applies_only_to_a_decision_that_would_recapture():
    history = _trace(*[_glare()] * 3)
    assert decide(_metrics(repair_targets=(TARGET,)), history).action == "repair"


def test_only_the_trailing_run_of_recaptures_is_counted():
    history = _trace(_glare(), _glare(), _metrics(repair_targets=(TARGET,)), _glare())
    assert decide(_glare(), history).action == "recapture"


def test_fixed_mm_lists_prior_targets_with_no_current_target_nearby():
    prior = _metrics(repair_targets=(TARGET, OTHER_TARGET))
    near_other = (OTHER_TARGET[0] + 7.0, OTHER_TARGET[1] - 5.0, 800.0)
    d = decide(_metrics(repair_targets=(near_other,)), _trace(prior))
    assert d.action == "repair"
    assert d.fixed_mm == ((TARGET[0], TARGET[1]),)


def test_fixed_mm_lists_every_prior_target_once_the_pane_passes():
    prior = _metrics(repair_targets=(TARGET, OTHER_TARGET))
    d = decide(_metrics(), _trace(prior))
    assert (d.action, d.fixed_mm) == ("pass", ((130.0, 130.0), (230.0, 230.0)))


def test_a_prior_target_now_hidden_as_a_possible_gap_is_not_called_fixed():
    # Glare over the old spot leaves a possible gap there; the other gap still needs repair.
    prior = _trace(_metrics(repair_targets=(TARGET, OTHER_TARGET)))
    hidden = _metrics(repair_targets=(OTHER_TARGET,), possible_centroids_mm=((TARGET[0] + 4.0, TARGET[1]),))
    d = decide(hidden, prior)
    assert d.action == "repair"
    assert d.fixed_mm == ()
    assert "repaired spot" not in d.message


def test_a_current_target_decides_fixed_at_the_match_distance():
    prior = _trace(_metrics(repair_targets=(TARGET,)))
    inside = (TARGET[0] + RULE_MM - 1.0, TARGET[1], 800.0)
    outside = (TARGET[0] + RULE_MM + 1.0, TARGET[1], 800.0)
    assert decide(_metrics(repair_targets=(inside,)), prior).fixed_mm == ()
    assert decide(_metrics(repair_targets=(outside,)), prior).fixed_mm == ((130.0, 130.0),)


def test_fixed_mm_is_empty_without_an_earlier_repair():
    assert decide(_metrics(), _trace(_glare())).fixed_mm == ()


def test_record_computes_the_effect_against_the_previous_record():
    first = _rejected("board_not_found")
    second = _metrics(unknown_fraction=0.2, largest_gap_mm=74.0, repair_targets=(TARGET,))
    trace = _trace(first, second)
    assert trace[0].effect == ()
    assert trace[1].step == 1
    assert trace[1].effect == (
        ("unknown_fraction", 1.0, 0.2),
        ("largest_gap_mm", 0.0, 74.0),
        ("repair_targets", 0.0, 1.0),
    )


def _five_step_trace():
    return _trace(
        _rejected("board_not_found"),
        _glare(frames=1),
        _metrics(frames=2, unknown_fraction=0.04, possible_centroids_mm=((120.0, 140.0),)),
        _metrics(frames=3, unknown_fraction=0.02, repair_targets=(TARGET,), largest_gap_mm=74.0),
        _metrics(frames=1, unknown_fraction=0.02),
    )


def test_a_five_step_trace_replays_to_its_recorded_decisions():
    trace = _five_step_trace()
    assert [r.decision.rule for r in trace] == [
        "board_not_found",
        "unknown_area",
        "possible_gap",
        "confirmed_gap",
        "all_clear",
    ]
    assert replay(trace) == [r.decision for r in trace]


def test_the_trace_survives_a_jsonl_round_trip():
    trace = _five_step_trace()
    text = trace_to_jsonl(trace)
    assert len(text.splitlines()) == len(trace)
    parsed = trace_from_jsonl(text)
    assert parsed == trace
    assert replay(parsed) == [r.decision for r in trace]


def test_an_empty_trace_round_trips():
    assert trace_from_jsonl(trace_to_jsonl([])) == []


def _every_decision():
    cases = [_rejected(reason) for reason in REJECT_REASONS] + [
        _metrics(reproj_rms_mm=2.0),
        _metrics(view_angle_deg=60.0),
        _metrics(native_mm_per_px=2.0),
        _glare(),
        _metrics(possible_centroids_mm=((120.0, 140.0),)),
        _metrics(repair_targets=(TARGET, OTHER_TARGET), largest_gap_mm=74.0),
        _metrics(),
    ]
    return [decide(metrics, []) for metrics in cases] + [_trace(*[_glare()] * 4)[-1].decision]


def test_every_rule_has_its_own_message_and_none_claims_the_window_is_safe():
    decisions = _every_decision()
    assert len({d.rule for d in decisions}) == len(decisions) == 12
    assert len({d.message for d in decisions}) == len(decisions)
    for d in decisions:
        lowered = d.message.lower()
        assert "safe" not in lowered and "compliant" not in lowered, d.message


def test_the_unknown_area_message_gives_the_percent_and_where_to_aim():
    message = decide(_glare(), []).message
    assert "20%" in message
    assert "150 mm across and 80 mm down from the top-left corner of the area checked" in message
    assert "sideways" in message


def test_the_too_oblique_message_gives_the_measured_angle():
    assert "60 degrees" in decide(_metrics(view_angle_deg=60.0), []).message


def test_the_confirmed_gap_message_gives_the_count_the_largest_gap_and_the_limit():
    d = decide(_metrics(repair_targets=(OTHER_TARGET, TARGET), largest_gap_mm=74.0), [])
    assert "2 gaps" in d.message
    assert "74 mm" in d.message
    assert "2 inch (51 mm)" in d.message
    assert "130 mm across and 130 mm down from the top-left corner of the area checked" in d.message


def test_the_all_clear_message_asks_about_placement_and_names_the_unchecked_area():
    message = decide(_metrics(unknown_fraction=0.03), []).message
    assert "outside surface" in message
    assert "under the calibration sheet was not checked" in message
    assert "3%" in message


def test_the_loop_cap_message_gives_the_gap_count_and_the_unread_percent():
    message = _trace(*[_glare(repair_targets=(TARGET,))] * 4)[-1].decision.message
    assert "1 gap" in message
    assert "20%" in message
