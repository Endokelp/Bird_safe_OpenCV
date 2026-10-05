"""Rule table that turns capture metrics into the next action, and the trace that records it.

It reads numbers only, never an image or a mask. The same metrics and history always produce the
same decision, so a saved trace replays to the decisions it recorded.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from birdsafe.coverage import RULE_MM
from birdsafe.markers import DOT_MM

# ponytail: a guess; synthetic captures read 0.06 mm. Retune on development captures, never on
# held-out sessions.
MAX_REPROJ_RMS_MM = 1.0
# Capture protocol limit.
MAX_VIEW_ANGLE_DEG = 45.0
# Four capture pixels across a dot.
MAX_NATIVE_MM_PER_PX = DOT_MM / 4
# ponytail: a guess at how much unreadable glass a pass can tolerate. Retune on development
# captures, never on held-out sessions.
MAX_UNKNOWN_FRACTION = 0.05
# ponytail: a guess at the smallest drop that counts as progress; sensor noise could exceed it.
# Retune on development captures, never on held-out sessions.
MIN_UNKNOWN_GAIN = 0.01
MAX_STALLED_RECAPTURES = 3
# A prior repair target still fails if a current target lies this close to it.
FIX_MATCH_MM = RULE_MM

_CAPTURE_MESSAGES = {
    "board_not_found": (
        "The calibration sheet was not found. "
        "Keep the whole sheet in frame and well lit, then retake the photo."
    ),
    "board_partly_hidden": (
        "Part of the calibration sheet is hidden. Clear anything covering it and keep all of it in frame."
    ),
    "view_too_oblique": (
        "The view is too steep to measure. Stand in front of the pane and aim the phone straight at it."
    ),
    "view_too_large": (
        "The photo spans too wide an area to measure. Move closer and aim the phone straight at the pane."
    ),
    "board_not_flat": (
        "The calibration sheet may be bent or the photo blurred. "
        "Press the sheet flat on the glass, hold the phone steady, and retake the photo."
    ),
    "too_oblique": (
        "The phone is {angle:.0f} degrees off straight on, over the {limit:.0f} degree limit. "
        "Move in front of the pane and aim straight at it."
    ),
    "too_far": "The pane is too far away to resolve the dots. Move closer and retake the photo.",
}


@dataclass(frozen=True)
class CaptureMetrics:
    # This capture. The three measurements are None when rectify rejected it.
    reject_reason: str | None
    reproj_rms_mm: float | None
    view_angle_deg: float | None
    native_mm_per_px: float | None
    # Fused state after this capture, unchanged from the last record if the capture was not fused.
    frames: int
    unknown_fraction: float
    unknown_centroid_mm: tuple[float, float] | None
    possible_centroids_mm: tuple[tuple[float, float], ...]
    repair_targets: tuple[tuple[float, float, float], ...]  # (x_mm, y_mm, area_mm2)
    largest_gap_mm: float


@dataclass(frozen=True)
class Decision:
    action: str  # "recapture" | "repair" | "pass" | "stop"
    rule: str
    message: str  # one or two plain sentences for the person holding the phone
    aim_mm: tuple[float, float] | None = None
    targets_mm: tuple[tuple[float, float, float], ...] = ()
    fixed_mm: tuple[tuple[float, float], ...] = ()  # earlier repair targets that now pass


@dataclass(frozen=True)
class TraceRecord:
    step: int
    metrics: CaptureMetrics
    decision: Decision
    # (metric, before, after) against the previous record. Empty on step 0.
    effect: tuple[tuple[str, float, float], ...]


def capture_rule(metrics: CaptureMetrics) -> str | None:
    if metrics.reject_reason is not None:
        return metrics.reject_reason
    if metrics.reproj_rms_mm is not None and metrics.reproj_rms_mm > MAX_REPROJ_RMS_MM:
        return "board_not_flat"
    if metrics.view_angle_deg is not None and metrics.view_angle_deg > MAX_VIEW_ANGLE_DEG:
        return "too_oblique"
    if metrics.native_mm_per_px is not None and metrics.native_mm_per_px > MAX_NATIVE_MM_PER_PX:
        return "too_far"
    return None


def _where(point: tuple[float, float]) -> str:
    x, y = point
    return f"{x:.0f} mm across and {y:.0f} mm down from the pane's top-left corner"


def _percent(fraction: float) -> str:
    return f"{fraction * 100:.0f}%"


def _gaps(count: int) -> str:
    return {0: "no gaps", 1: "1 gap"}.get(count, f"{count} gaps")


def _capture_message(rule: str, metrics: CaptureMetrics) -> str:
    return _CAPTURE_MESSAGES[rule].format(angle=metrics.view_angle_deg, limit=MAX_VIEW_ANGLE_DEG)


def _unknown_message(metrics: CaptureMetrics) -> str:
    centred = f", centred {_where(metrics.unknown_centroid_mm)}" if metrics.unknown_centroid_mm else ""
    return (
        f"{_percent(metrics.unknown_fraction)} of the pane could not be read{centred}. "
        "Step sideways so the glare moves off it, then retake the photo."
    )


def _possible_gap_message(aim: tuple[float, float]) -> str:
    return (
        f"A gap wider than 2 inches may be hiding {_where(aim)}, but that spot could not be read. "
        "Step sideways and photograph it again."
    )


def _confirmed_gap_message(metrics: CaptureMetrics) -> str:
    x, y, _ = max(metrics.repair_targets, key=lambda target: target[2])
    return (
        f"Found {_gaps(len(metrics.repair_targets))} wider than the 2 inch ({RULE_MM:.0f} mm) limit; "
        f"the largest is {metrics.largest_gap_mm:.0f} mm. "
        f"Add markings {_where((x, y))}."
    )


def _all_clear_message(metrics: CaptureMetrics) -> str:
    return (
        f"No gap wider than 2 inches was found in the area checked "
        f"({_percent(metrics.unknown_fraction)} could not be read). "
        "Confirm the markings are on the outside surface of the glass; "
        "the area under the calibration sheet was not checked."
    )


def _loop_cap_message(metrics: CaptureMetrics) -> str:
    return (
        "Stopping here, because more photos are not improving the result. "
        f"Found {_gaps(len(metrics.repair_targets))} wider than 2 inches, and "
        f"{_percent(metrics.unknown_fraction)} of the pane could not be read."
    )


def _fixed_note(fixed: tuple[tuple[float, float], ...]) -> str:
    if not fixed:
        return ""
    return "The repaired spot now passes. " if len(fixed) == 1 else f"{len(fixed)} repaired spots now pass. "


def _fixed_targets(
    metrics: CaptureMetrics, history: Sequence[TraceRecord]
) -> tuple[tuple[float, float], ...]:
    last_repair = next((r for r in reversed(history) if r.decision.action == "repair"), None)
    if last_repair is None:
        return ()
    return tuple(
        (x, y)
        for x, y, _ in last_repair.decision.targets_mm
        if not any(math.hypot(x - cx, y - cy) <= FIX_MATCH_MM for cx, cy, _ in metrics.repair_targets)
    )


def _first_match(metrics: CaptureMetrics, history: Sequence[TraceRecord]) -> Decision:
    rule = capture_rule(metrics)
    if rule is not None:
        return Decision("recapture", rule, _capture_message(rule, metrics))
    if metrics.unknown_fraction > MAX_UNKNOWN_FRACTION:
        return Decision(
            "recapture", "unknown_area", _unknown_message(metrics), aim_mm=metrics.unknown_centroid_mm
        )
    if metrics.possible_centroids_mm and not metrics.repair_targets:
        aim = metrics.possible_centroids_mm[0]
        return Decision("recapture", "possible_gap", _possible_gap_message(aim), aim_mm=aim)
    fixed = _fixed_targets(metrics, history)
    if metrics.repair_targets:
        return Decision(
            "repair",
            "confirmed_gap",
            _fixed_note(fixed) + _confirmed_gap_message(metrics),
            targets_mm=metrics.repair_targets,
            fixed_mm=fixed,
        )
    return Decision(
        "pass", "all_clear", _fixed_note(fixed) + _all_clear_message(metrics), fixed_mm=fixed
    )


def _stalled(before: CaptureMetrics, after: CaptureMetrics) -> bool:
    gained = before.unknown_fraction - after.unknown_fraction >= MIN_UNKNOWN_GAIN
    recovered = capture_rule(before) is not None and capture_rule(after) is None
    return not (gained or recovered)


def _stalled_recaptures(metrics: CaptureMetrics, history: Sequence[TraceRecord]) -> int:
    # Each record is judged by the capture that followed it, which for the last record is the
    # capture being decided now.
    after = [r.metrics for r in history[1:]] + [metrics]
    count = 0
    for prior, next_metrics in zip(reversed(history), reversed(after)):
        if prior.decision.action != "recapture":
            break
        count += _stalled(prior.metrics, next_metrics)
    return count


def decide(metrics: CaptureMetrics, history: Sequence[TraceRecord]) -> Decision:
    decision = _first_match(metrics, history)
    if (
        decision.action == "recapture"
        and _stalled_recaptures(metrics, history) >= MAX_STALLED_RECAPTURES
    ):
        return Decision(
            "stop", "loop_cap", _loop_cap_message(metrics), targets_mm=metrics.repair_targets
        )
    return decision


def _effect(before: CaptureMetrics, after: CaptureMetrics) -> tuple[tuple[str, float, float], ...]:
    return (
        ("unknown_fraction", before.unknown_fraction, after.unknown_fraction),
        ("largest_gap_mm", before.largest_gap_mm, after.largest_gap_mm),
        ("repair_targets", float(len(before.repair_targets)), float(len(after.repair_targets))),
    )


def record(metrics: CaptureMetrics, history: Sequence[TraceRecord]) -> TraceRecord:
    effect = _effect(history[-1].metrics, metrics) if history else ()
    return TraceRecord(len(history), metrics, decide(metrics, history), effect)


def replay(records: Sequence[TraceRecord]) -> list[Decision]:
    return [decide(r.metrics, records[:i]) for i, r in enumerate(records)]


def trace_to_jsonl(records: Sequence[TraceRecord]) -> str:
    return "".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in records)


def _tuples(rows) -> tuple[tuple, ...]:
    return tuple(tuple(row) for row in rows)


def _optional_tuple(row) -> tuple | None:
    return None if row is None else tuple(row)


def _metrics_from_dict(data: dict) -> CaptureMetrics:
    return CaptureMetrics(
        **{
            **data,
            "unknown_centroid_mm": _optional_tuple(data["unknown_centroid_mm"]),
            "possible_centroids_mm": _tuples(data["possible_centroids_mm"]),
            "repair_targets": _tuples(data["repair_targets"]),
        }
    )


def _decision_from_dict(data: dict) -> Decision:
    return Decision(
        **{
            **data,
            "aim_mm": _optional_tuple(data["aim_mm"]),
            "targets_mm": _tuples(data["targets_mm"]),
            "fixed_mm": _tuples(data["fixed_mm"]),
        }
    )


def trace_from_jsonl(text: str) -> list[TraceRecord]:
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        records.append(
            TraceRecord(
                data["step"],
                _metrics_from_dict(data["metrics"]),
                _decision_from_dict(data["decision"]),
                _tuples(data["effect"]),
            )
        )
    return records
