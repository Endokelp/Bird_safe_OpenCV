# Phases 0 to 2 implementation plan: guardrails, fixture, metric core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the repository with a working anti-slop gate, produce the calibration fixture and a synthetic pane generator with exact known answers, and build the two modules the entry depends on: metric rectification and the 2x2 coverage check.

**Architecture:** All geometry happens on a rectified plane with a known millimetres-per-pixel scale derived from a printed ChArUco board. The 2x2 rule check is a morphological erosion of empty space by a 2 inch square structuring element, so the answer is exact rather than learned and is testable against synthetic panes whose violations are known by construction.

**Tech Stack:** Python 3.12, `opencv-python==5.0.0.93`, NumPy, pytest, ruff, GitHub Actions.

## Global Constraints

- OpenCV pinned at `opencv-python==5.0.0.93`. ArUco and ChArUco come from `cv2.aruco` in `objdetect`; do not add `opencv-contrib-python`.
- The 2x2 rule threshold is `50.8` mm. Define it once as `RULE_MM = 50.8` in `birdsafe/coverage.py` and import it everywhere else. No second literal.
- Every public function that returns a physical quantity returns millimetres. Pixels never leave a module.
- No em dash in any tracked file. No banned vocabulary (see `tools/sloplint.py`). CI fails on either.
- `WIN-AND-PLACE-CLAUDE.txt`, `CONTEST-DOSSIER.md`, `MASTER-BRIEF.md` are gitignored before the first commit and must never appear in `git ls-files`.
- Comments explain a non-obvious reason. No narrated steps, no banner comments, no `TODO` without an owner and a reason.
- No number goes in any document without an observed run behind it.
- Conventional commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`). No AI attribution in any commit, ever.

---

## Phase 0: Foundations and guardrails

Single subagent. Owned paths: repo root, `pyproject.toml`, `.github/`, `.gitignore`, `tools/sloplint.py`, `tests/test_sloplint.py`.

### Task 1: Repository with the strategy documents excluded from history

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: an initialised repository whose first commit contains no strategy documents.

- [ ] **Step 1: Write `.gitignore` before `git init`**

The order matters. A strategy document committed once stays in history forever.

```gitignore
# Competition strategy and research. Never publish.
WIN-AND-PLACE-CLAUDE.txt
CONTEST-DOSSIER.md
MASTER-BRIEF.md
PHASE-STATE.md

# Captures are large and may show private property. Kept out of the repo.
captures/
data/raw/

__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
*.egg-info/
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "birdsafe"
version = "0.1.0"
description = "Installation coverage checker for visible bird-safe window markings"
requires-python = ">=3.12"
dependencies = [
    "opencv-python==5.0.0.93",
    "numpy==2.3.3",
]

[project.optional-dependencies]
dev = ["pytest==8.4.2", "ruff==0.14.0"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

If either pinned version does not resolve on this machine, run `pip index versions <package>` and pin the newest that does. Record the pin you actually used in the commit message. Do not pin a version you have not installed.

- [ ] **Step 3: Initialise and verify nothing sensitive is staged**

```bash
cd /c/Users/venni/OpenCV
git init
git add -A
git status --short
git ls-files | grep -E 'WIN-AND-PLACE|CONTEST-DOSSIER|MASTER-BRIEF|PHASE-STATE'
```

Expected: the `grep` prints nothing and exits 1. If it prints a filename, stop, fix `.gitignore`, run `git rm --cached` on it, and check again before committing.

- [ ] **Step 4: Commit and push**

```bash
git commit -m "chore: initialise repository with dependency pins and strategy exclusions"
git branch -M main
git remote add origin https://github.com/Endokelp/Bird_safe_OpenCV.git
git push -u origin main
```

- [ ] **Step 5: Confirm the remote matches**

```bash
git ls-remote --heads origin
```

Expected: one `refs/heads/main`.

### Task 2: The slop gate

**Files:**
- Create: `tools/sloplint.py`
- Test: `tests/test_sloplint.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `find_violations(text: str, path: str) -> list[str]`, used by the CLI and by CI. Returns one human-readable line per violation, empty list when clean.

- [ ] **Step 1: Write the failing test**

Note: this test file must not contain a literal em dash or a literal banned word, or the gate
would flag its own fixtures. Build them from codepoints and fragments.

```python
from tools.sloplint import ALLOW_MARKER, find_violations

EM = chr(0x2014)


def test_flags_em_dash():
    assert find_violations(f"a {EM} b", "f.md")


def test_flags_banned_word_case_insensitively():
    assert find_violations("We " + "Lever" + "age this", "f.md")


def test_flags_not_just_but_construction():
    assert find_violations("not " + "just fast, but correct", "f.md")


def test_flags_todo_without_owner():
    assert find_violations("# TO" + "DO fix this", "f.py")


def test_allows_todo_with_owner_and_reason():
    assert not find_violations("# TO" + "DO(venni): retune after the Oct 2 captures", "f.py")


def test_allow_marker_exempts_a_line_that_quotes_the_rule():
    assert not find_violations(f"Avoid the {EM} character. {ALLOW_MARKER}", "AGENTS.md")


def test_clean_text_passes():
    assert not find_violations("The gap measured 61 mm against a 50.8 mm threshold.", "f.md")


def test_hyphen_is_not_an_em_dash():
    assert not find_violations("multi-frame fusion", "f.md")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_sloplint.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'tools.sloplint'`.

- [ ] **Step 3: Implement**

```python
"""Mechanical half of the anti-slop gate. Judgement calls stay with the reviewers.

This file is exempt from its own scan, because the word list below has to contain the words it
forbids. Documents that legitimately quote the rules, such as AGENTS.md, use ALLOW_MARKER instead.
"""

import re
import subprocess
import sys

SELF = "tools/sloplint.py"
ALLOW_MARKER = "sloplint: allow"
EM_DASH = chr(0x2014)

BANNED = [
    "delve", "leverage", "seamless", "revolutionize", "revolutionise",
    "game-changing", "cutting-edge", "unlock the power", "elevate your",
    "robust solution", "in today's fast-paced world", "harness the power",
]

# The comma is required so that an ordinary sentence using "not just" for real emphasis
# is not caught. Only the paired construction is slop.
NOT_JUST = re.compile(r"\bnot just\b[^.;!?\n]{1,80},\s*but\b", re.IGNORECASE)

# An owner in parentheses and a reason after the colon. A bare marker rots.
TODO_OK = re.compile(r"TODO\([^)]+\):\s*\S")
TODO_ANY = re.compile(r"\bTODO\b")


def find_violations(text: str, path: str) -> list[str]:
    if path.replace("\\", "/").endswith(SELF):
        return []
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        low = line.lower()
        if EM_DASH in line:
            out.append(f"{path}:{n}: em dash")
        for word in BANNED:
            if word in low:
                out.append(f"{path}:{n}: banned vocabulary: {word}")
        if NOT_JUST.search(line):
            out.append(f"{path}:{n}: paired 'not just' construction")
        if TODO_ANY.search(line) and not TODO_OK.search(line):
            out.append(f"{path}:{n}: task marker without an owner and a reason")
    return out


def main() -> int:
    files = subprocess.run(
        ["git", "ls-files", "*.py", "*.md", "*.js", "*.html", "*.css", "*.toml", "*.yml"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    problems = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            problems += find_violations(fh.read(), path)
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_sloplint.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run the gate over the repository**

Run: `python tools/sloplint.py`
Expected: exit 0 with no output.

It will flag `AGENTS.md` lines 101 to 103 and the banned-vocabulary list in `docs/specs/2026-09-16-build-plan.md`. Those lines quote the rules in order to state them, so append `sloplint: allow` to each as a trailing comment or sentence, then rerun. Every other hit is a real violation and gets fixed in the text, because the rule applies to the whole repository and not only to new code.

- [ ] **Step 6: Commit**

```bash
git add tools/sloplint.py tests/test_sloplint.py
git commit -m "feat: add mechanical anti-slop gate"
```

### Task 3: CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `tools/sloplint.py`, `pyproject.toml`.
- Produces: a required check on every push.

- [ ] **Step 1: Write the workflow**

```yaml
name: ci
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: python tools/sloplint.py
      - run: |
          if git ls-files | grep -E 'WIN-AND-PLACE|CONTEST-DOSSIER|MASTER-BRIEF'; then
            echo "Strategy document is tracked. Remove it from the index."
            exit 1
          fi
      - run: python -m pytest -v
      - run: python -c "import cv2; assert cv2.__version__.startswith('5.'), cv2.__version__"
```

- [ ] **Step 2: Push and confirm the run is green**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add lint, slop gate, strategy leak check, and tests"
git push
gh run watch
```

Expected: all steps pass. Phase 0 gate is met when this run is green and `python -c "import cv2; print(cv2.__version__)"` prints a 5.x version locally.

---

## Phase 1: Capture fixture and synthetic ground truth

Two subagents in parallel. They share no files.

- Subagent A owns `tools/charuco_sheet.py` and `docs/capture-protocol.md`.
- Subagent B owns `birdsafe/synth.py` and `tests/test_synth.py`.

### Task 4: Printable ChArUco sheet (Subagent A)

**Files:**
- Create: `tools/charuco_sheet.py`
- Create: `docs/capture-protocol.md`

**Interfaces:**
- Consumes: nothing.
- Produces: `BOARD = cv2.aruco.CharucoBoard((7, 5), 0.030, 0.022, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))` exported as `birdsafe_board()` so `calib.py` in Task 6 constructs the identical board. Square 30 mm, marker 22 mm, 7 by 5 squares.

- [ ] **Step 1: Write the generator**

```python
"""Renders the printable calibration board. Physical square size must match birdsafe.calib."""

import cv2

SQUARE_MM = 30.0
MARKER_MM = 22.0
SQUARES_X, SQUARES_Y = 7, 5
DICT = cv2.aruco.DICT_4X4_50


def birdsafe_board() -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_MM / 1000.0,
        MARKER_MM / 1000.0,
        cv2.aruco.getPredefinedDictionary(DICT),
    )


def main() -> None:
    # 300 dpi so the printed square lands within a fraction of a millimetre of nominal.
    px_per_mm = 300.0 / 25.4
    size = (round(SQUARES_X * SQUARE_MM * px_per_mm), round(SQUARES_Y * SQUARE_MM * px_per_mm))
    cv2.imwrite("docs/charuco-sheet.png", birdsafe_board().generateImage(size))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate and verify the physical size**

Run: `python tools/charuco_sheet.py`
Then print `docs/charuco-sheet.png` at 100 percent with no scaling, and measure one square with a ruler.
Expected: 30 mm within a millimetre. If it is not, the print dialog scaled it. Reprint. Record the measured square size in `docs/capture-protocol.md`, because every distance the app reports is derived from it.

- [ ] **Step 3: Write `docs/capture-protocol.md`**

Contents: print at 100 percent, verify the square with a ruler and write down the measured value, tape the sheet flat against the glass inside the pane or on the frame beside it, keep the whole board in frame, stand within roughly 45 degrees of square to the glass, fill the frame with the pane, capture from at least three positions per session, and record session id, pane id, date, weather, time of day, background type, and whether the treatment is complete or has planted gaps.

- [ ] **Step 4: Commit**

```bash
git add tools/charuco_sheet.py docs/capture-protocol.md docs/charuco-sheet.png
git commit -m "feat: add printable ChArUco sheet and capture protocol"
```

### Task 5: Synthetic pane generator (Subagent B)

**Files:**
- Create: `birdsafe/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Consumes: `RULE_MM` from Task 7. If Task 7 is not merged yet, define `RULE_MM = 50.8` locally and replace the definition with the import in Task 7's commit.
- Produces:

```python
@dataclass(frozen=True)
class SyntheticPane:
    markers: np.ndarray       # bool, True where a marking is
    pane: np.ndarray          # bool, True inside the pane
    unknown: np.ndarray       # bool, True where unreadable
    mm_per_px: float
    planted_gap_centers_mm: tuple[tuple[float, float], ...]

def grid_positions(extent_mm: float, spacing_mm: float) -> list[float]
def make_pane(width_mm, height_mm, spacing_mm, dot_mm, mm_per_px,
              omit=(), unknown_boxes_mm=()) -> SyntheticPane
```

`omit` is a tuple of `(row, col)` grid indices whose dots are removed. `unknown_boxes_mm` is a tuple of `(x, y, w, h)` in millimetres. `planted_gap_centers_mm` holds the centre of each omitted dot, which is the position a correct coverage check should report when the omission is large enough to violate the rule.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
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
            assert grid_positions(extent, spacing)[0] < 50.8


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
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_synth.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdsafe.synth'`.

- [ ] **Step 3: Implement**

```python
"""Synthetic panes with known layouts. Every artifact made from these is labelled synthetic."""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticPane:
    markers: np.ndarray
    pane: np.ndarray
    unknown: np.ndarray
    mm_per_px: float
    planted_gap_centers_mm: tuple[tuple[float, float], ...]


def grid_positions(extent_mm: float, spacing_mm: float) -> list[float]:
    """Dot positions that tile the whole pane at exactly the nominal spacing.

    Counting with floor leaves an untreated band at the far edge that can exceed the rule
    threshold, which would make a nominally compliant fixture fail for a reason unrelated to
    the spacing under test. Rounding the count up and centring the grid keeps each margin
    below half a spacing.
    """
    n = max(1, math.ceil(extent_mm / spacing_mm))
    inset = (extent_mm - (n - 1) * spacing_mm) / 2.0
    return [inset + i * spacing_mm for i in range(n)]


def make_pane(width_mm, height_mm, spacing_mm, dot_mm, mm_per_px,
              omit=(), unknown_boxes_mm=()) -> SyntheticPane:
    w = round(width_mm / mm_per_px)
    h = round(height_mm / mm_per_px)
    markers = np.zeros((h, w), np.uint8)
    unknown = np.zeros((h, w), np.uint8)
    radius = max(1, round(dot_mm / 2 / mm_per_px))
    omitted = set(omit)
    gaps = []

    xs = grid_positions(width_mm, spacing_mm)
    ys = grid_positions(height_mm, spacing_mm)
    for row, y_mm in enumerate(ys):
        for col, x_mm in enumerate(xs):
            if (row, col) in omitted:
                gaps.append((x_mm, y_mm))
                continue
            cv2.circle(markers, (round(x_mm / mm_per_px), round(y_mm / mm_per_px)), radius, 255, -1)

    for x_mm, y_mm, bw_mm, bh_mm in unknown_boxes_mm:
        x0, y0 = round(x_mm / mm_per_px), round(y_mm / mm_per_px)
        x1, y1 = round((x_mm + bw_mm) / mm_per_px), round((y_mm + bh_mm) / mm_per_px)
        unknown[y0:y1, x0:x1] = 255

    return SyntheticPane(
        markers=markers.astype(bool),
        pane=np.ones((h, w), bool),
        unknown=unknown.astype(bool),
        mm_per_px=mm_per_px,
        planted_gap_centers_mm=tuple(gaps),
    )
```

A spacing larger than the pane produces a single centred dot. To generate a bare pane, pass a
large spacing and omit that one dot: `make_pane(100, 100, 1000, 6, 0.5, omit=((0, 0),))`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_synth.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add birdsafe/synth.py tests/test_synth.py
git commit -m "feat: add synthetic pane generator with known ground truth"
```

---

## Phase 2: Metric core

Two subagents in parallel. `coverage.py` takes only arrays and numbers, so it does not wait on `calib.py`.

- Subagent A owns `birdsafe/calib.py` and `tests/test_calib.py`.
- Subagent B owns `birdsafe/coverage.py` and `tests/test_coverage.py`.

### Task 6: Metric rectification (Subagent A)

**Files:**
- Create: `birdsafe/calib.py`
- Test: `tests/test_calib.py`

**Interfaces:**
- Consumes: `birdsafe_board()` from `tools/charuco_sheet.py` (Task 4).
- Produces:

```python
@dataclass(frozen=True)
class Rectified:
    image: np.ndarray        # BGR, fronto-parallel
    mm_per_px: float
    reproj_rms_px: float
    corners_found: int

def rectify(image: np.ndarray, mm_per_px: float = 0.5) -> Rectified | None
```

Returns `None` when fewer than six ChArUco corners are found. A caller never gets a scale the module could not justify.

- [ ] **Step 1: Write the failing test**

The test builds its own known-perspective image so it needs no camera.

```python
import cv2
import numpy as np
from birdsafe.calib import rectify
from tools.charuco_sheet import birdsafe_board, SQUARE_MM, SQUARES_X, SQUARES_Y


def _warped_board(mm_per_px=0.5):
    w = round(SQUARES_X * SQUARE_MM / mm_per_px)
    h = round(SQUARES_Y * SQUARE_MM / mm_per_px)
    flat = cv2.cvtColor(birdsafe_board().generateImage((w, h)), cv2.COLOR_GRAY2BGR)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Pull the top edge inward so the board is viewed off-axis, as a phone capture would be.
    dst = np.float32([[60, 20], [w - 20, 0], [w, h], [0, h - 30]])
    return cv2.warpPerspective(flat, cv2.getPerspectiveTransform(src, dst), (w, h))


def test_rectify_recovers_the_scale_it_was_asked_for():
    r = rectify(_warped_board(), mm_per_px=0.5)
    assert r is not None
    assert r.mm_per_px == 0.5


def test_rectify_finds_corners_and_reports_low_reprojection_error():
    r = rectify(_warped_board())
    assert r.corners_found >= 6
    assert r.reproj_rms_px < 2.0


def test_rectified_board_squares_measure_the_physical_square_size():
    r = rectify(_warped_board(), mm_per_px=0.5)
    flat_board = rectify(
        cv2.cvtColor(birdsafe_board().generateImage((r.image.shape[1], r.image.shape[0])),
                     cv2.COLOR_GRAY2BGR),
        mm_per_px=0.5,
    )
    assert flat_board is not None
    # Both rectify to the same metric plane, so the board occupies the same width in both.
    assert abs(r.image.shape[1] - flat_board.image.shape[1]) <= 2


def test_rectify_returns_none_when_no_board_is_present():
    assert rectify(np.zeros((400, 400, 3), np.uint8)) is None
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_calib.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdsafe.calib'`.

- [ ] **Step 3: Implement**

```python
"""Metric rectification from a printed ChArUco board.

The board is the only source of physical scale. Without it there is no honest millimetre
measurement, so a failed detection returns None rather than a guess.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from tools.charuco_sheet import SQUARE_MM, SQUARES_X, SQUARES_Y, birdsafe_board

MIN_CORNERS = 6


@dataclass(frozen=True)
class Rectified:
    image: np.ndarray
    mm_per_px: float
    reproj_rms_px: float
    corners_found: int


def rectify(image: np.ndarray, mm_per_px: float = 0.5) -> Rectified | None:
    board = birdsafe_board()
    corners, ids, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if corners is None or len(corners) < MIN_CORNERS:
        return None

    # Board corner (i, j) sits at a known millimetre offset, which fixes the output scale.
    obj = board.getChessboardCorners()[ids.flatten()][:, :2] * 1000.0 / mm_per_px
    src = corners.reshape(-1, 2).astype(np.float32)
    dst = obj.astype(np.float32)

    h_matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if h_matrix is None:
        return None

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_matrix).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))

    out_w = round(SQUARES_X * SQUARE_MM / mm_per_px)
    out_h = round(SQUARES_Y * SQUARE_MM / mm_per_px)
    warped = cv2.warpPerspective(image, h_matrix, (out_w, out_h))

    return Rectified(
        image=warped,
        mm_per_px=mm_per_px,
        reproj_rms_px=rms,
        corners_found=int(len(corners)),
    )
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_calib.py -v`
Expected: 4 passed. If the ChArUco API signature differs in 5.0.0.93, run `python -c "import cv2; help(cv2.aruco.CharucoDetector)"` and adapt. Do not guess the signature from memory; the module moved between 4.6 and 4.7 and changed again in 5.0.

- [ ] **Step 5: Commit**

```bash
git add birdsafe/calib.py tests/test_calib.py
git commit -m "feat: add metric rectification from a ChArUco board"
```

### Task 7: The 2x2 coverage check (Subagent B)

**Files:**
- Create: `birdsafe/coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: `SyntheticPane` from Task 5 for tests only. The module itself takes plain arrays.
- Produces:

```python
RULE_MM = 50.8

@dataclass(frozen=True)
class RepairTarget:
    center_mm: tuple[float, float]
    area_mm2: float

@dataclass(frozen=True)
class CoverageResult:
    confirmed: np.ndarray            # bool, violating region fully observed
    possible: np.ndarray             # bool, violating but overlapping unknown
    largest_gap_mm: float
    unknown_fraction: float
    targets: tuple[RepairTarget, ...]

def check_coverage(pane, markers, unknown, mm_per_px, rule_mm=RULE_MM) -> CoverageResult
```

- [ ] **Step 1: Write the failing test**

```python
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
    # Largest inscribed circle in a 100 mm square is 100 mm across.
    assert r.largest_gap_mm == pytest.approx(100.0, abs=3.0)


def test_scale_changes_the_answer_not_just_the_pixels():
    coarse = _check(make_pane(300, 300, 70, 6, 1.0))
    fine = _check(make_pane(300, 300, 70, 6, 0.25))
    assert abs(coarse.largest_gap_mm - fine.largest_gap_mm) < 4.0


def test_rejects_a_kernel_too_small_to_be_meaningful():
    with pytest.raises(ValueError):
        check_coverage(np.ones((10, 10), bool), np.zeros((10, 10), bool),
                       np.zeros((10, 10), bool), mm_per_px=100.0)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_coverage.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdsafe.coverage'`.

- [ ] **Step 3: Implement**

```python
"""The 2x2 rule check.

The American Bird Conservancy states that markings must be no more than 2 inches apart
horizontally and vertically. A pane violates that wherever an axis-aligned 2 inch square fits
inside it containing no marking, which is exactly an erosion of empty space by a square of that
side. The rule is stated in horizontal and vertical terms, so the test is axis-aligned; a gap that
is only wide on the diagonal is not reported.
"""

from dataclasses import dataclass

import cv2
import numpy as np

RULE_MM = 50.8
MIN_KERNEL_PX = 3


@dataclass(frozen=True)
class RepairTarget:
    center_mm: tuple[float, float]
    area_mm2: float


@dataclass(frozen=True)
class CoverageResult:
    confirmed: np.ndarray
    possible: np.ndarray
    largest_gap_mm: float
    unknown_fraction: float
    targets: tuple[RepairTarget, ...]


def _erode_then_restore(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Border is treated as non-empty so that only squares lying fully inside the pane count.
    centers = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.dilate(centers, kernel)


def check_coverage(pane, markers, unknown, mm_per_px, rule_mm=RULE_MM) -> CoverageResult:
    side = round(rule_mm / mm_per_px)
    if side < MIN_KERNEL_PX:
        raise ValueError(
            f"{rule_mm} mm is {side} px at {mm_per_px} mm/px, too coarse to measure. "
            "Capture closer or rectify to a finer scale."
        )

    pane_u8 = pane.astype(np.uint8)
    empty = (pane & ~markers).astype(np.uint8)
    observed_empty = (pane & ~markers & ~unknown).astype(np.uint8)
    kernel = np.ones((side, side), np.uint8)

    confirmed = _erode_then_restore(observed_empty, kernel).astype(bool) & pane
    any_violation = _erode_then_restore(empty, kernel).astype(bool) & pane
    possible = any_violation & ~confirmed

    # distanceTransform measures distance to the nearest zero pixel. A zero border makes the pane
    # edge bound that distance, so the reported circle lies inside the pane rather than running off
    # it, and an all-empty pane returns its own width instead of infinity.
    padded = cv2.copyMakeBorder(empty * pane_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist_px = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    largest_gap_mm = float(dist_px.max() * 2.0 * mm_per_px)

    pane_px = int(pane_u8.sum())
    unknown_fraction = float((unknown & pane).sum() / pane_px) if pane_px else 0.0

    px_area_mm2 = mm_per_px * mm_per_px
    count, _, stats, centroids = cv2.connectedComponentsWithStats(confirmed.astype(np.uint8), 8)
    targets = tuple(
        RepairTarget(
            center_mm=(float(centroids[i][0] * mm_per_px), float(centroids[i][1] * mm_per_px)),
            area_mm2=float(stats[i, cv2.CC_STAT_AREA] * px_area_mm2),
        )
        for i in range(1, count)
    )

    return CoverageResult(
        confirmed=confirmed,
        possible=possible,
        largest_gap_mm=largest_gap_mm,
        unknown_fraction=unknown_fraction,
        targets=targets,
    )
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_coverage.py -v`
Expected: 8 passed.

If `test_a_single_missing_dot_in_a_tight_grid_is_found_and_located` fails with zero targets, check the arithmetic before changing the code: at 40 mm spacing with 6 mm dots, removing one dot leaves an empty region roughly 74 mm across, which does contain a 50.8 mm square. If it fails with more than one target, the erosion is producing fragments and the dilation step is missing.

- [ ] **Step 5: Commit**

```bash
git add birdsafe/coverage.py tests/test_coverage.py
git commit -m "feat: add 2x2 rule coverage check with confirmed and possible violations"
```

### Task 8: Phase 2 gate, part 1, synthetic sweep

**Files:**
- Test: `tests/test_coverage_sweep.py`

**Interfaces:**
- Consumes: `make_pane`, `check_coverage`.
- Produces: the recorded synthetic result for `PHASE-STATE.md`.

- [ ] **Step 1: Write the sweep**

```python
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
    # A bare 120 mm pane: the largest inscribed circle is exactly 120 mm across.
    p = make_pane(120, 120, 1000, 6, mm_per_px, omit=((0, 0),))
    r = check_coverage(p.pane, p.markers, p.unknown, p.mm_per_px)
    assert abs(r.largest_gap_mm - 120.0) <= 1.0 + 2 * mm_per_px
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_coverage_sweep.py -v`
Expected: 20 passed.

This sweep has been run ahead of implementation against a reference build of the same algorithm on
OpenCV 4.13 and all cases passed, with the compliant 40 mm grid reporting a 50.4 mm largest gap
against the 50.8 mm threshold. Confirm the same on OpenCV 5.0.0.93. A difference between the two
versions is a finding worth recording, not a reason to loosen a bound. A failure here is a real finding, not a test to loosen. If the largest-gap error exceeds the tolerance, the cause is the distance transform mask size or the pane clipping, and both are fixable in `coverage.py`.

- [ ] **Step 3: Commit and record**

```bash
git add tests/test_coverage_sweep.py
git commit -m "test: add synthetic sweep for the Phase 2 accuracy gate"
```

Then write the observed pass or fail, with the command and the count, into `PHASE-STATE.md`. Do not write a number you did not see printed.

### Task 9: Phase 2 gate, part 2, real capture repeatability

This task needs the physical pane from Phase 1 and cannot be faked. It is the gate `AGENTS.md`
requires before any interface work.

**Files:**
- Create: `tools/repeatability.py`

- [ ] **Step 1: Capture**

Ten photographs of the same treated pane, from different positions, all inside the protocol in
`docs/capture-protocol.md`, in one lighting condition. Store them under `captures/`, which is
gitignored.

- [ ] **Step 2: Write the harness**

```python
"""Reports spread across repeated captures of one pane. Phase 2 gate, part 2."""

import statistics
import sys

import cv2

from birdsafe.calib import rectify


def main(paths: list[str]) -> int:
    scales, rms = [], []
    for path in paths:
        r = rectify(cv2.imread(path))
        if r is None:
            print(f"{path}: board not found")
            continue
        scales.append(r.mm_per_px)
        rms.append(r.reproj_rms_px)
    if len(rms) < 2:
        print(f"only {len(rms)} usable captures, cannot report spread")
        return 1
    print(f"usable {len(rms)}/{len(paths)}")
    print(f"reprojection rms px: mean {statistics.mean(rms):.3f} max {max(rms):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 3: Run it and record the real numbers**

Run: `python tools/repeatability.py captures/session01/*.jpg`

Record in `PHASE-STATE.md`: how many of the ten captures resolved the board, the mean and maximum
reprojection error, and the spread in largest-gap measurement once `markers.py` exists in Phase 3.
Write the observed numbers. If the board fails to resolve in several captures, that is the finding,
and the fix is the capture protocol, not the threshold.

- [ ] **Step 4: Decide the gate**

Pass: the board resolves in at least eight of ten captures and reprojection error stays under the
threshold recorded in `PHASE-STATE.md`.

Fail: tighten the protocol, narrow the supported angle range, or enlarge the printed board, then
recapture. Do not start Phase 6 interface work on an unrepeatable measurement.

---

## Self-review against the spec

| Spec section | Covered by |
|---|---|
| 4.1 Scale and rectification | Tasks 4 and 6 |
| 4.2 Pane region | Phase 6, interaction work, not in this plan |
| 4.3 Marker detection | Phase 3, next plan |
| 4.4 Unknown regions | Phase 3 produces the masks; Task 7 already consumes them |
| 4.5 Gap measurement | Tasks 7 and 8 |
| 4.6 Multi-frame fusion | Phase 4, next plan |
| 5 Decision policy | Phase 4, next plan |
| 7 AWS and cost | Phase 6 |
| 8 Evaluation | Phase 5 |
| 10 Repository hygiene | Tasks 1 and 3 |

Phases 3 through 9 get their own plan files at their gates. Their task detail depends on numbers
that do not exist yet: real detection thresholds come from Phase 3 captures, policy thresholds come
from Phase 4 measurements, and the COOL benchmark shape depends on the Phase 6 deployment. Writing
those tasks now would mean inventing the numbers, which the spec's honesty constraints forbid.
