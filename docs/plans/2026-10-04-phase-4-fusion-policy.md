# Phase 4 implementation plan: fusion, decision policy, and trace

**Goal:** Turn single-capture analysis into a scan loop. Several captures of one pane fuse onto the
board plane, a deterministic rule table reads the fused metrics and picks the next action
(recapture with a reason and an aim point, repair with millimetre targets, pass, or stop), and every
decision is written to a trace that replays to the same decisions. The trace is the Agentic Vision
evidence. Built and tested on synthetic captures; the real multi-frame gate waits for captures.

**Spec:** `docs/specs/2026-09-16-coverage-checker-spec.md` sections 4.6 and 5.

**Tech stack:** Python 3.12, `opencv-python==5.0.0.93`, NumPy, pytest, ruff. No new dependencies.

## Global constraints

- All Phase 3 global constraints still apply (`docs/plans/2026-09-16-phase-3-detection.md`):
  `RULE_MM` only in `coverage.py`, `DOT_MM` only in `markers.py`, millimetres for every physical
  quantity, no em dash, sloplint clean, comments explain a non-obvious reason, conventional commits,
  no AI attribution, work from the repo root, edit only owned paths.
- Run checks as `.venv/Scripts/python -m pytest -q`, `.venv/Scripts/ruff check .`,
  `.venv/Scripts/python tools/sloplint.py`.
- `policy.py` reads numbers and returns decisions. It never touches an image or a mask.
- Thresholds that are guesses until development captures exist carry a `ponytail:` comment naming
  the ceiling: retune on development captures, never on held-out sessions.
- Do not change detection or quality floors. They stay frozen until development captures exist.

## Design decisions

1. **Fused markers come only from resolved pixels.** A marker detected under glare in one frame
   must not hide a gap that another frame resolved as empty (the Phase 2 regression). Per frame,
   `resolved = seen & ~unknown`; fused markers are the union of `markers & resolved`; fused unknown
   is canvas pixels that no frame resolved.
2. **Pixels outside the warped capture are unknown.** `warpPerspective` fills them with black, which
   the camera never saw. `rectify` returns a `valid` mask and the caller ORs `~valid` into unknown.
3. **The calibration sheet is not under test.** It is taped on the glass inside the pane in every
   frame, so its paper footprint is removed from the region under test. Treating it as unknown would
   make the policy ask for recaptures that can never succeed. The report must say this area is not
   checked.
4. **Capture checks run before fusion.** A rejected, blurred-board, too-oblique, or too-coarse
   capture is not fused. Its trace record carries the unchanged fused state, so the log never shows
   a metric that a discarded frame produced.
5. **A repair starts a new set of frames.** After a `repair` decision the pane has physically
   changed, so earlier frames are stale. The trace continues across the reset.
6. **Coordinates are from the canvas top-left.** The canvas is the region under test in the board
   frame. Until the Phase 6 pane quadrilateral exists, callers pass the pane rectangle.

## File map

| Path | Role | Task | Owner |
|---|---|---|---|
| `birdsafe/calib.py` | `Rejected`, `valid`, `native_mm_per_px`, `view_angle_deg` | 1 | Sonnet |
| `tests/test_calib.py` | Update `is None` checks, test new fields | 1 | Sonnet |
| `tools/repeatability.py` | Print reject reason, OR `~valid` into unknown | 1 | Sonnet |
| `tools/charuco_sheet.py` | `sheet_footprint_mm()` | 1 | Sonnet |
| `tests/test_charuco_sheet.py` | Footprint test | 1 | Sonnet |
| `birdsafe/fusion.py` | `Observation`, `FusedMasks`, `fuse` | 2 | Sonnet |
| `tests/test_fusion.py` | Fusion tests | 2 | Sonnet |
| `birdsafe/policy.py` | Metrics, decisions, rule table, trace, replay, JSONL | 3 | Grok |
| `tests/test_policy.py` | One test per rule row, cap, replay | 3 | Grok |
| `birdsafe/scan.py` | `ScanState`, `scan_step` | 4 | Sonnet |
| `tests/test_scan.py` | Synthetic glare, repair, rescan loop and replay | 4 | Sonnet |

Tasks 1 to 3 run in parallel. Task 4 starts after all three pass.

---

### Task 1: Calibration outputs the policy needs

**Produces** (in `birdsafe/calib.py`):

```python
@dataclass(frozen=True)
class Rectified:
    image: np.ndarray
    mm_per_px: float
    board_origin_mm: tuple[float, float]
    reproj_rms_mm: float
    corners_found: int
    valid: np.ndarray          # bool, image.shape[:2]; True where the pixel came from the capture
    native_mm_per_px: float    # capture resolution on the board plane, coarsest axis
    view_angle_deg: float      # off-axis angle from foreshortening at the board centre

@dataclass(frozen=True)
class Rejected:
    reason: str

def rectify(image, mm_per_px=0.5) -> Rectified | Rejected
```

Reason codes, one per existing `return None`:

| Existing reject | `reason` |
|---|---|
| no corners, too few corners, `findHomography` returns None | `board_not_found` |
| corners all in one row or one column | `board_partly_hidden` |
| `ref_w == 0`, or a frame corner on the wrong side of the horizon | `view_too_oblique` |
| output side above `MAX_SIDE_PX` | `view_too_large` |

- `valid`: warp a 255-filled `uint8` image of the capture's size with the same matrix and
  `cv2.INTER_NEAREST`, then `> 0`.
- `native_mm_per_px` and `view_angle_deg`: take the board centre in output pixels (mean of `dst`),
  step 1 mm along x and along y, map the three points back with the inverse homography (including
  the translate), and form the 2x2 Jacobian J in source pixels per mm. With singular values
  `s1 >= s2`: `native_mm_per_px = 1 / s2`, `view_angle_deg = degrees(arccos(min(1, s2 / s1)))`.
  Comment: without camera intrinsics this is a foreshortening estimate, not a measured pose.
- Remove the `ponytail:` note about returning a reason, since this task does it.

**Also:**

- `tools/repeatability.py`: on `Rejected`, print `f"{path}: {rectified.reason}"`. Pass
  `quality.unknown | ~rectified.valid` to `check_coverage`.
- `tools/charuco_sheet.py`: add `sheet_footprint_mm() -> tuple[float, float, float, float]`
  returning `(x0, y0, x1, y1)` of the printed page in the board frame (origin at the board's
  top-left square corner, y down). Derive it from the same layout numbers `sheet_pdf` uses; move
  that arithmetic into one helper both functions call so the page layout lives in one place.
  Expected about `(-34.7, -27.0, 244.7, 189.0)`.

**Tests:**

- Every existing `is None` assertion becomes an `isinstance(r, Rejected)` check plus the expected
  `reason`; `is not None` becomes `isinstance(r, Rectified)`.
- A square-on view (all-zero offsets in `_warped_scene`) gives `view_angle_deg < 5` and
  `native_mm_per_px` within 10% of 0.5. `VIEW_A` gives a larger angle than the square-on view.
- `VIEW_A`: `valid` is not all True, more than half True, and True at the board centre.
- Footprint: page size is 279.4 x 215.9 mm within 0.1 mm and the page contains the 210 x 150 mm
  board.

### Task 2: Fusion

**Produces** (`birdsafe/fusion.py`, NumPy only):

```python
@dataclass(frozen=True)
class Observation:
    markers: np.ndarray                  # bool, rectified frame grid
    unknown: np.ndarray                  # bool, already includes ~valid
    board_origin_mm: tuple[float, float] # from Rectified
    mm_per_px: float

@dataclass(frozen=True)
class FusedMasks:
    markers: np.ndarray   # bool, canvas grid
    unknown: np.ndarray   # bool, canvas grid
    frames: int

def fuse(observations, canvas_mm) -> FusedMasks
```

- `canvas_mm = (x_mm, y_mm, width_mm, height_mm)` in the board frame. Canvas pixel (0, 0) sits at
  board-frame `(x_mm, y_mm)`. Shape `(round(height_mm / mpp), round(width_mm / mpp))`.
- Observation pixel `(u, v)` is board-frame `(u * mpp - ox, v * mpp - oy)` with
  `(ox, oy) = board_origin_mm`. So canvas `(cu, cv)` reads observation
  `(cu + round((x_mm + ox) / mpp), cv + round((y_mm + oy) / mpp))`. The rounding is at most half a
  pixel; say so in one comment.
- Canvas pixels outside an observation's extent are not seen by it.
- `resolved = seen & ~unknown`. Fused markers: OR of `markers & resolved`. Fused unknown: NOT (OR of
  `resolved`).
- Empty `observations`: all-unknown canvas, no markers, `frames=0`. Mixed `mm_per_px`: `ValueError`.

**Tests:**

- One observation aligned with the canvas reproduces `markers & ~unknown` and `unknown`.
- Two observations with glare boxes in different places: fused unknown is their overlap, and the
  unknown fraction after two frames is below the fraction after one.
- A marker inside frame A's unknown area, where frame B resolved the same pixels as empty, is not a
  fused marker.
- An observation offset by a known `board_origin_mm` lands on the expected canvas pixels; canvas
  pixels beyond its extent stay unknown.
- Mixed `mm_per_px` raises `ValueError`.

### Task 3: Decision policy and trace

**Produces** (`birdsafe/policy.py`, standard library only, plus `DOT_MM` import):

```python
@dataclass(frozen=True)
class CaptureMetrics:
    # This capture. All None when rectify rejected it.
    reject_reason: str | None
    reproj_rms_mm: float | None
    view_angle_deg: float | None
    native_mm_per_px: float | None
    # Fused state after this capture (unchanged from the last record if the capture was not fused).
    frames: int
    unknown_fraction: float
    unknown_centroid_mm: tuple[float, float] | None
    possible_centroids_mm: tuple[tuple[float, float], ...]
    repair_targets: tuple[tuple[float, float, float], ...]   # (x_mm, y_mm, area_mm2)
    largest_gap_mm: float

@dataclass(frozen=True)
class Decision:
    action: str     # "recapture" | "repair" | "pass" | "stop"
    rule: str       # rule that fired, see table
    message: str    # one plain sentence for the person holding the phone
    aim_mm: tuple[float, float] | None = None
    targets_mm: tuple[tuple[float, float, float], ...] = ()
    fixed_mm: tuple[tuple[float, float], ...] = ()   # prior repair targets that now pass

@dataclass(frozen=True)
class TraceRecord:
    step: int
    metrics: CaptureMetrics
    decision: Decision
    # (metric, before, after) against the previous record: unknown_fraction, largest_gap_mm,
    # repair_targets (count). Empty on step 0.
    effect: tuple[tuple[str, float, float], ...]

def capture_rule(metrics: CaptureMetrics) -> str | None
def decide(metrics: CaptureMetrics, history: Sequence[TraceRecord]) -> Decision
def record(metrics: CaptureMetrics, history: Sequence[TraceRecord]) -> TraceRecord
def replay(records: Sequence[TraceRecord]) -> list[Decision]
def trace_to_jsonl(records) -> str
def trace_from_jsonl(text: str) -> list[TraceRecord]
```

Constants (provisional ones carry a `ponytail:` comment):

| Name | Value | Basis |
|---|---|---|
| `MAX_REPROJ_RMS_MM` | 1.0 | provisional; synthetic reads 0.06 |
| `MAX_VIEW_ANGLE_DEG` | 45.0 | capture protocol |
| `MAX_NATIVE_MM_PER_PX` | `DOT_MM / 4` | four capture pixels across a dot |
| `MAX_UNKNOWN_FRACTION` | 0.05 | provisional |
| `MIN_UNKNOWN_GAIN` | 0.01 | provisional |
| `MAX_STALLED_RECAPTURES` | 3 | spec 5 loop cap |
| `FIX_MATCH_MM` | `RULE_MM` | a prior target is still failing if a current target is within this |

`capture_rule` returns the first of these that fires, else None: `reject_reason` itself (when
not None), `board_not_flat` (RMS above max), `too_oblique` (angle above max), `too_far` (native
mm/px above max). `scan.py` calls it to decide whether to fuse; `decide` calls it first.

`decide`, first match wins:

| Order | Condition | action | rule | aim / targets |
|---|---|---|---|---|
| 1 | `capture_rule` fires | recapture | that rule | none |
| 2 | `unknown_fraction > MAX_UNKNOWN_FRACTION` | recapture | `unknown_area` | `aim_mm = unknown_centroid_mm` |
| 3 | possible centroids present, no repair targets | recapture | `possible_gap` | `aim_mm` = first possible centroid |
| 4 | repair targets present | repair | `confirmed_gap` | `targets_mm` = repair targets |
| 5 | otherwise | pass | `all_clear` | none |

- Loop cap: if the decision would be `recapture`, count trailing consecutive `recapture` records in
  `history`. Each one counts as stalled when the capture after it (the next record, or the current
  metrics for the last one) did not lower `unknown_fraction` by at least `MIN_UNKNOWN_GAIN` and did
  not turn a capture-rule failure into a pass. At `MAX_STALLED_RECAPTURES` stalled, return
  `action="stop"`, `rule="loop_cap"`, with `targets_mm` set to any repair targets, and a message
  that states how many gaps were found and what percent of the pane could not be read.
- `fixed_mm`, on `repair` and `pass`: find the most recent `repair` decision in `history`. Each of
  its targets is fixed when no current repair target lies within `FIX_MATCH_MM` of it.
- `pass` message asks the person to confirm the markings are on the outside surface and says the
  area under the calibration sheet was not checked. No message claims the window is bird-safe or
  compliant.
- Messages are one or two short plain sentences with numbers rounded for a person (whole mm,
  whole percent), coordinates stated as "from the pane's top-left corner".
- `record` sets `step = len(history)` and computes `effect` against `history[-1].metrics`.
- `replay` returns `decide(r.metrics, records[:i])` for each record.
- JSONL: one record per line via `dataclasses.asdict` and `json.dumps(sort_keys=True)`. Parsing
  converts lists back to tuples so a round-tripped trace compares equal and replays identically.

**Tests:** one per rule row and per capture rule, precedence (a capture failure beats a gap),
loop cap fires on the third stalled recapture and not before, an improving recapture in the run is
not counted as stalled, `fixed_mm` after a repair, JSONL round trip equality, and replay of a
hand-built five-step trace matching its recorded decisions.

### Task 4: Scan step and the synthetic loop

**Produces** (`birdsafe/scan.py`):

```python
@dataclass(frozen=True)
class ScanState:
    canvas_mm: tuple[float, float, float, float]
    mm_per_px: float = 0.5
    observations: tuple[Observation, ...] = ()
    trace: tuple[TraceRecord, ...] = ()

def scan_step(state: ScanState, capture: Rectified | Rejected) -> ScanState
```

1. If the last trace decision is `repair`, drop all observations before using this capture.
2. Build the capture fields of `CaptureMetrics` from `capture` (all None plus the reason when
   `Rejected`). Call `capture_rule` on metrics that carry the current fused state.
3. If no capture rule fires: `detect_markers` and `unknown_regions` on `capture.image`, then append
   `Observation(markers.mask, quality.unknown | ~capture.valid, capture.board_origin_mm,
   capture.mm_per_px)`.
4. Fuse, build the region under test (canvas minus `sheet_footprint_mm()` mapped into canvas
   pixels), run `check_coverage(region, fused.markers, fused.unknown, mm_per_px)`.
5. `unknown_centroid_mm`: centroid of the largest connected component of `fused.unknown & region`.
   `possible_centroids_mm`: component centroids of `coverage.possible`, largest first.
   `repair_targets`: from `coverage.targets`.
6. Return a new state with the record from `policy.record` appended.

**Tests** (`tests/test_scan.py`), with synthetic `Rectified` objects built from `render_pane` and
the canvas placed right of the sheet (`canvas_mm = (300, 0, 300, 300)`, `board_origin_mm =
(-300, 0)`, `valid` all True, square-on capture fields):

- Glare over the planted gap, then glare elsewhere, then the repaired pane:
  `unknown_area` recapture aimed within 20 mm of the glare centre; then `repair` with one target
  within 10 mm of the planted gap and a lower unknown fraction than step 0; then `pass` with that
  target in `fixed_mm`.
- The trace round-trips through JSONL and `replay` reproduces every recorded decision.
- A capture with `view_angle_deg=60` gives `too_oblique` and leaves `observations` unchanged.
- `Rejected("board_not_found")` gives a `board_not_found` recapture.
- The same glare four times stops with `loop_cap`.
- A canvas overlapping the sheet footprint excludes it: an unmarked sheet area yields no repair
  target.

## Gate

- Synthetic (this plan): all tests pass, ruff and sloplint clean, unknown fraction falls as frames
  are added, a trace replays to the same decisions.
- Real (blocked on captures): unknown fraction falls across a real multi-frame session, and that
  session's trace replays.
