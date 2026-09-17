# Phase 3 implementation plan: marker detection and quality masks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect visible dot markings and unreadable (glare or blur) pixels on a rectified pane so the existing 2x2 coverage check can run on real-looking images, using synthetic rendered captures until physical footage exists.

**Architecture:** `render_pane` turns a `SyntheticPane` layout into a BGR image with known dots, optional texture, planted glare, and optional blur. `detect_markers` resizes to a fixed working scale, isolates either-polarity blobs with top-hat plus adaptive Gaussian threshold, and keeps contours whose physical diameter and circularity match a 6 mm dot. `unknown_regions` marks near-saturated low-texture pixels as glare and low Laplacian energy as too blurry to resolve a dot. Coverage is unchanged: it still consumes masks, never images.

**Tech Stack:** Python 3.12, `opencv-python==5.0.0.93`, NumPy, pytest, ruff. No new dependencies.

## Global Constraints

- OpenCV pinned at `opencv-python==5.0.0.93`. Do not add `opencv-contrib-python`.
- `RULE_MM = 50.8` lives only in `birdsafe/coverage.py`. Import it. Do not write a second 50.8.
- `DOT_MM = 6.0` lives only in `birdsafe/markers.py`. `quality.py` and tests import it. Do not write a second 6.0 default diameter.
- Every public function that returns a physical quantity returns millimetres. Pixel masks stay in the image grid because `check_coverage` already consumes masks.
- No em dash in any tracked file. No banned vocabulary (see `tools/sloplint.py`).
- Comments explain a non-obvious reason. No narrated steps, no banner comments, no `TODO` without an owner and a reason. (sloplint: allow)
- No number goes in any document without an observed run behind it.
- Threshold constants used on synthetic renders are labelled with a `ponytail:` comment naming the ceiling: retune on development captures, never on held-out sessions.
- Conventional commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`). No AI attribution in any commit, ever.
- Work from the repository root. Run tests as `.venv/Scripts/python -m pytest <file> -v`. Run ruff as `.venv/Scripts/ruff check .` before each commit. Tools that import `birdsafe` run as modules: `python -m tools.repeatability ...`.
- Do not edit files outside the task's owned paths. If you need a change elsewhere, stop and report it.
- Do not tune on, or claim results from, real phone captures. Those do not exist yet.

## File map

| Path | Role | Owner task |
|---|---|---|
| `birdsafe/synth.py` | Add `render_pane` | 1 |
| `tests/test_synth.py` | Render tests | 1 |
| `birdsafe/markers.py` | `DOT_MM`, `detect_markers` | 2 |
| `tests/test_markers.py` | Detection precision/recall | 2 |
| `birdsafe/quality.py` | `unknown_regions` | 3 |
| `tests/test_quality.py` | Glare and blur masks | 3 |
| `tests/test_detect_pipeline.py` | Render to coverage on a planted gap | 4 |
| `tools/repeatability.py` | Also report `largest_gap_mm` spread | 4 |

---

### Task 1: Rendered synthetic captures

**Files:**
- Modify: `birdsafe/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Consumes: existing `SyntheticPane` from `make_pane` (markers, pane, unknown, mm_per_px, planted_gap_centers_mm). Do not change that dataclass.
- Produces:

```python
def render_pane(
    pane: SyntheticPane,
    *,
    dark_on_light: bool = True,
    light: int = 220,
    dark: int = 40,
    texture_std: float = 8.0,
    glare_boxes_mm: tuple = (),
    blur_sigma_px: float = 0.0,
    rng_seed: int = 0,
) -> np.ndarray:
```

Returns a BGR `uint8` image with the same height and width as `pane.markers`. `glare_boxes_mm` is a tuple of `(x_mm, y_mm, width_mm, height_mm)` filled with 255 after the dots are drawn, so glare can hide markings. Blur is applied last. `texture_std` is Gaussian noise on the background before dots, so quality tests can tell sharp empty glass from blur.

- [ ] **Step 1: Write the failing tests at the end of `tests/test_synth.py`**

Keep the existing tests and imports. Add `import cv2` next to the existing imports, then append:

```python
def test_render_matches_the_marker_mask_shape_and_is_bgr():
    p = make_pane(300, 300, 40, 6, 0.5)
    image = render_pane(p)
    assert image.shape == (p.markers.shape[0], p.markers.shape[1], 3)
    assert image.dtype == np.uint8


def test_dark_dots_are_darker_than_the_background():
    p = make_pane(300, 300, 40, 6, 0.5)
    image = render_pane(p, dark_on_light=True, texture_std=0.0)
    gray = image.mean(axis=2)
    assert gray[p.markers].mean() < gray[~p.markers].mean()


def test_light_dots_are_brighter_than_the_background():
    p = make_pane(300, 300, 40, 6, 0.5)
    image = render_pane(p, dark_on_light=False, texture_std=0.0)
    gray = image.mean(axis=2)
    assert gray[p.markers].mean() > gray[~p.markers].mean()


def test_glare_box_is_saturated_and_hides_dots():
    p = make_pane(300, 300, 40, 6, 0.5)
    image = render_pane(p, texture_std=0.0, glare_boxes_mm=((20.0, 20.0, 100.0, 100.0),))
    x0, y0 = round(20.0 / p.mm_per_px), round(20.0 / p.mm_per_px)
    x1, y1 = round(120.0 / p.mm_per_px), round(120.0 / p.mm_per_px)
    patch = image[y0 + 10:y1 - 10, x0 + 10:x1 - 10]
    assert patch.min() == 255


def test_blur_lowers_high_frequency_energy():
    p = make_pane(300, 300, 40, 6, 0.5)
    sharp = render_pane(p, texture_std=8.0, blur_sigma_px=0.0)
    blurry = render_pane(p, texture_std=8.0, blur_sigma_px=8.0)
    # Laplacian energy is the same measure quality.py uses to decide a region cannot resolve a dot.
    assert _laplacian_energy(blurry) < 0.01 * _laplacian_energy(sharp)


def _laplacian_energy(image):
    gray = image.mean(axis=2)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float((lap * lap).mean())
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_synth.py -v`

Expected: FAIL on `render_pane` import or the new tests. Existing six tests still pass.

- [ ] **Step 3: Implement `render_pane` in `birdsafe/synth.py`**

Add `render_pane` below `make_pane`. Do not change `make_pane` or `SyntheticPane`.

```python
def render_pane(
    pane,
    *,
    dark_on_light=True,
    light=220,
    dark=40,
    texture_std=8.0,
    glare_boxes_mm=(),
    blur_sigma_px=0.0,
    rng_seed=0,
):
    """BGR view of a synthetic pane. Labelled synthetic: layouts are constructed, not photographed."""
    height, width = pane.markers.shape
    background, foreground = (light, dark) if dark_on_light else (dark, light)
    gray = np.full((height, width), background, np.uint8)
    if texture_std > 0:
        noise = np.random.default_rng(rng_seed).normal(0.0, texture_std, (height, width))
        gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gray[pane.markers] = foreground
    for x_mm, y_mm, box_w_mm, box_h_mm in glare_boxes_mm:
        x0 = round(x_mm / pane.mm_per_px)
        y0 = round(y_mm / pane.mm_per_px)
        x1 = round((x_mm + box_w_mm) / pane.mm_per_px)
        y1 = round((y_mm + box_h_mm) / pane.mm_per_px)
        gray[y0:y1, x0:x1] = 255
    if blur_sigma_px > 0:
        ksize = max(3, int(round(blur_sigma_px * 6)) | 1)
        gray = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma_px)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_synth.py -v`

Expected: all tests in the file pass, including the original six.

- [ ] **Step 5: Commit**

```bash
git add birdsafe/synth.py tests/test_synth.py
git commit -m "feat: render synthetic pane captures with known glare and blur"
```

---

### Task 2: Marker detection

**Files:**
- Create: `birdsafe/markers.py`
- Test: `tests/test_markers.py`

**Interfaces:**
- Consumes: BGR or grayscale `uint8` image (rectified), `mm_per_px`, optional `expected_dot_mm`. Uses `render_pane` and `make_pane` only in tests.
- Produces:

```python
DOT_MM = 6.0
WORKING_MM_PER_PX = 0.5

@dataclass(frozen=True)
class MarkerResult:
    mask: np.ndarray  # bool, same height and width as the input image
    centers_mm: tuple[tuple[float, float], ...]  # from the array origin

def detect_markers(image, mm_per_px, expected_dot_mm=DOT_MM) -> MarkerResult
```

Detection chain, in order: grayscale, resize so millimetres-per-pixel equals `WORKING_MM_PER_PX`, Gaussian pre-blur, white top-hat and black top-hat, adaptive Gaussian threshold on each, union, `findContours`, filter by circularity and physical diameter, draw kept contours, resize the mask back with nearest-neighbour.

Constants (do not scatter literals):

```python
# ponytail: 1.2 px at 0.5 mm/px. Stops texture from fragmenting adaptive threshold.
# Retune if real 6 mm dots smear. Held-out captures stay sealed.
PRE_BLUR_SIGMA = 1.2
TOPHAT_SCALE = 1.6
ADAPTIVE_C = -2
MIN_CIRCULARITY = 0.6
DIAMETER_LO = 0.5
DIAMETER_HI = 2.0
```

- [ ] **Step 1: Write `tests/test_markers.py`**

```python
import math

import cv2
import numpy as np

from birdsafe.markers import DOT_MM, detect_markers
from birdsafe.synth import grid_positions, make_pane, render_pane


def _truth_centers_mm(width_mm, height_mm, spacing_mm, omit=()):
    omitted = set(omit)
    xs = grid_positions(width_mm, spacing_mm)
    ys = grid_positions(height_mm, spacing_mm)
    return [
        (x_mm, y_mm)
        for row, y_mm in enumerate(ys)
        for col, x_mm in enumerate(xs)
        if (row, col) not in omitted
    ]


def _match(detected, truth, tol_mm=DOT_MM):
    unused = list(detected)
    hits = 0
    for tx, ty in truth:
        best_i = None
        best_d = tol_mm
        for i, (dx, dy) in enumerate(unused):
            dist = math.hypot(dx - tx, dy - ty)
            if dist <= best_d:
                best_d = dist
                best_i = i
        if best_i is not None:
            hits += 1
            unused.pop(best_i)
    return hits, len(unused), len(truth) - hits


def _recall_case(dark_on_light, mm_per_px=0.5, omit=()):
    pane = make_pane(300, 300, 40, DOT_MM, mm_per_px, omit=omit)
    image = render_pane(pane, dark_on_light=dark_on_light)
    result = detect_markers(image, pane.mm_per_px)
    truth = _truth_centers_mm(300, 300, 40, omit)
    return _match(result.centers_mm, truth) + (result,)


def test_dark_dots_on_light_are_all_found():
    tp, fp, fn, result = _recall_case(True)
    assert fn == 0 and fp == 0 and tp == 64
    assert result.mask.shape[:2] == (600, 600)


def test_light_dots_on_dark_are_all_found():
    tp, fp, fn, _result = _recall_case(False)
    assert fn == 0 and fp == 0 and tp == 64


def test_an_omitted_dot_is_not_detected():
    tp, fp, fn, _result = _recall_case(True, omit=((2, 2),))
    assert fn == 0 and fp == 0 and tp == 63


def test_detection_survives_downscale_to_working_resolution():
    tp, fp, fn, _result = _recall_case(True, mm_per_px=0.25)
    assert fn == 0 and fp == 0 and tp == 64


def test_a_large_square_is_not_a_dot():
    pane = make_pane(300, 300, 40, DOT_MM, 0.5)
    image = render_pane(pane, texture_std=0.0)
    x0, y0 = round(20 / pane.mm_per_px), round(20 / pane.mm_per_px)
    side = round(40 / pane.mm_per_px)
    cv2.rectangle(image, (x0, y0), (x0 + side, y0 + side), (20, 20, 20), -1)
    result = detect_markers(image, pane.mm_per_px)
    square_center = (20 + 20, 20 + 20)
    near = [
        (x_mm, y_mm)
        for x_mm, y_mm in result.centers_mm
        if math.hypot(x_mm - square_center[0], y_mm - square_center[1]) < 15
    ]
    assert near == []
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_markers.py -v`

Expected: FAIL, `ModuleNotFoundError: No module named 'birdsafe.markers'`.

- [ ] **Step 3: Implement `birdsafe/markers.py`**

```python
"""Detect visible bird-safe dots on a rectified pane."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

DOT_MM = 6.0
WORKING_MM_PER_PX = 0.5
# ponytail: 1.2 px at 0.5 mm/px. Stops texture from fragmenting adaptive threshold.
# Retune if real 6 mm dots smear. Held-out captures stay sealed.
PRE_BLUR_SIGMA = 1.2
TOPHAT_SCALE = 1.6
ADAPTIVE_C = -2
MIN_CIRCULARITY = 0.6
DIAMETER_LO = 0.5
DIAMETER_HI = 2.0


@dataclass(frozen=True)
class MarkerResult:
    mask: np.ndarray
    centers_mm: tuple[tuple[float, float], ...]


def _odd_ksize(value: float) -> int:
    return max(3, int(round(value)) | 1)


def detect_markers(image, mm_per_px, expected_dot_mm=DOT_MM) -> MarkerResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = mm_per_px / WORKING_MM_PER_PX
    src_h, src_w = gray.shape[:2]
    work_h = max(1, round(src_h * scale))
    work_w = max(1, round(src_w * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    working = cv2.resize(gray, (work_w, work_h), interpolation=interpolation)
    working = cv2.GaussianBlur(
        working, (_odd_ksize(PRE_BLUR_SIGMA * 6), _odd_ksize(PRE_BLUR_SIGMA * 6)), PRE_BLUR_SIGMA
    )

    dot_px = expected_dot_mm / WORKING_MM_PER_PX
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_ksize(TOPHAT_SCALE * dot_px),) * 2)
    block = _odd_ksize(dot_px)
    union = np.zeros((work_h, work_w), np.uint8)
    for morph in (cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT):
        response = cv2.morphologyEx(working, morph, se)
        union |= cv2.adaptiveThreshold(
            response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, ADAPTIVE_C
        )

    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_work = np.zeros((work_h, work_w), np.uint8)
    centers = []
    lo_mm = DIAMETER_LO * expected_dot_mm
    hi_mm = DIAMETER_HI * expected_dot_mm
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 1 or perimeter < 1:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        diameter_mm = 2 * math.sqrt(area / math.pi) * WORKING_MM_PER_PX
        if circularity < MIN_CIRCULARITY or not (lo_mm <= diameter_mm <= hi_mm):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        centers.append((cx * WORKING_MM_PER_PX, cy * WORKING_MM_PER_PX))
        cv2.drawContours(mask_work, [contour], -1, 255, -1)

    if (work_h, work_w) != (src_h, src_w):
        mask = cv2.resize(mask_work, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_work
    return MarkerResult(mask=mask.astype(bool), centers_mm=tuple(centers))
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_markers.py tests/test_synth.py -v`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdsafe/markers.py tests/test_markers.py
git commit -m "feat: detect either-polarity dots by physical diameter"
```

---

### Task 3: Unknown regions from glare and blur

**Files:**
- Create: `birdsafe/quality.py`
- Test: `tests/test_quality.py`

**Interfaces:**
- Consumes: BGR or grayscale `uint8` image, `mm_per_px`, optional `expected_dot_mm` (default `DOT_MM` imported from `birdsafe.markers`).
- Produces:

```python
SATURATION = 250
# ponytail: floors measured on synthetic texture_std=8. Retune on development captures.
TEXTURE_STD = 8.0
BLUR_ENERGY_FLOOR = 200.0
WINDOW_SCALE = 2.0

@dataclass(frozen=True)
class QualityResult:
    unknown: np.ndarray  # bool, same shape as the input
    glare: np.ndarray
    blur: np.ndarray

def unknown_regions(image, mm_per_px, expected_dot_mm=DOT_MM) -> QualityResult
```

Glare: value >= `SATURATION` and local standard deviation below `TEXTURE_STD`. Blur: local mean of Laplacian-squared below `BLUR_ENERGY_FLOOR`. Window side is `WINDOW_SCALE * expected_dot_mm / mm_per_px`, odd, at least 3. `unknown = glare OR blur`. Do not call `detect_markers`.

- [ ] **Step 1: Write `tests/test_quality.py`**

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_quality.py -v`

Expected: FAIL, `ModuleNotFoundError: No module named 'birdsafe.quality'`.

- [ ] **Step 3: Implement `birdsafe/quality.py`**

```python
"""Unreadable pixels: glare and blur that hide the glass surface."""

from dataclasses import dataclass

import cv2
import numpy as np

from birdsafe.markers import DOT_MM

SATURATION = 250
# ponytail: floors measured on synthetic texture_std=8 (sharp Laplacian energy ~3800,
# blurred ~4, glare interior std 0). Retune on development captures. Held-out stays sealed.
TEXTURE_STD = 8.0
BLUR_ENERGY_FLOOR = 200.0
WINDOW_SCALE = 2.0


@dataclass(frozen=True)
class QualityResult:
    unknown: np.ndarray
    glare: np.ndarray
    blur: np.ndarray


def unknown_regions(image, mm_per_px, expected_dot_mm=DOT_MM) -> QualityResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    window = max(3, int(round(WINDOW_SCALE * expected_dot_mm / mm_per_px)) | 1)
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (window, window))
    mean_sq = cv2.blur(gray_f * gray_f, (window, window))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    glare = (gray >= SATURATION) & (std < TEXTURE_STD)

    laplacian = cv2.Laplacian(gray_f, cv2.CV_64F, ksize=3)
    energy = cv2.blur(laplacian * laplacian, (window, window))
    blur = energy < BLUR_ENERGY_FLOOR
    unknown = glare | blur
    return QualityResult(unknown=unknown, glare=glare, blur=blur)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_quality.py tests/test_markers.py tests/test_synth.py -v`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdsafe/quality.py tests/test_quality.py
git commit -m "feat: mark glare and unresolved blur as unknown"
```

---

### Task 4: Synthetic pipeline and repeatability gap spread

**Files:**
- Create: `tests/test_detect_pipeline.py`
- Modify: `tools/repeatability.py`

**Interfaces:**
- Consumes: `render_pane`, `detect_markers`, `unknown_regions`, `check_coverage`.
- Produces: a test that a planted missing dot on a rendered pane becomes a confirmed coverage target, and a glare-hidden missing dot becomes a possible target. `repeatability.py` still reports reprojection RMS; it also reports `largest_gap_mm` mean and max when detection succeeds. Captures that rectify but then fail detection still count as usable for RMS and are skipped for the gap statistic.

- [ ] **Step 1: Write `tests/test_detect_pipeline.py`**

```python
import cv2

from birdsafe.coverage import RULE_MM, check_coverage
from birdsafe.markers import DOT_MM, detect_markers
from birdsafe.quality import unknown_regions
from birdsafe.synth import make_pane, render_pane


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
```

- [ ] **Step 2: Run the pipeline tests and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_detect_pipeline.py -v`

Expected: FAIL until the file exists; after the file exists they should PASS if Tasks 2 and 3 are correct. If they fail on behaviour, stop and report; do not weaken the assertions.

- [ ] **Step 3: Extend `tools/repeatability.py`**

Replace the file with:

```python
"""Reports spread across repeated captures of one pane. Phase 2 gate, part 2.

Run from the repository root as a module so birdsafe and tools import:
`python -m tools.repeatability captures/session01/*.jpg`
"""

import statistics
import sys

import cv2
import numpy as np

from birdsafe.calib import rectify
from birdsafe.coverage import check_coverage
from birdsafe.markers import detect_markers
from birdsafe.quality import unknown_regions


def main(paths: list[str]) -> int:
    rms = []
    gaps = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"{path}: unreadable")
            continue
        rectified = rectify(image)
        if rectified is None:
            print(f"{path}: no usable board view (not found, too oblique, or partly hidden)")
            continue
        rms.append(rectified.reproj_rms_mm)
        markers = detect_markers(rectified.image, rectified.mm_per_px)
        quality = unknown_regions(rectified.image, rectified.mm_per_px)
        # ponytail: no pane quadrilateral until Phase 6. Repeatability measures the
        # full rectified frame, which is enough to compare captures of one taped sheet.
        pane = np.ones(markers.mask.shape, bool)
        coverage = check_coverage(pane, markers.mask, quality.unknown, rectified.mm_per_px)
        gaps.append(coverage.largest_gap_mm)
    if len(rms) < 2:
        print(f"only {len(rms)} usable captures, cannot report spread")
        return 1
    print(f"usable {len(rms)}/{len(paths)}")
    print(f"reprojection rms mm: mean {statistics.mean(rms):.3f} max {max(rms):.3f}")
    if len(gaps) >= 2:
        print(f"largest gap mm: mean {statistics.mean(gaps):.3f} max {max(gaps):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Add a focused test for the gap statistic at the bottom of `tests/test_detect_pipeline.py`**

```python
import numpy as np

from tools.repeatability import main as repeatability_main


def test_repeatability_reports_gap_spread_on_two_usable_views(tmp_path, capsys):
    from birdsafe.calib import rectify
    from tests.test_calib import _warped_scene, VIEW_A, VIEW_B

    paths = []
    for i, view in enumerate((VIEW_A, VIEW_B)):
        path = tmp_path / f"view{i}.png"
        cv2.imwrite(str(path), _warped_scene(view))
        paths.append(str(path))
        assert rectify(_warped_scene(view)) is not None

    assert repeatability_main(paths) == 0
    out = capsys.readouterr().out
    assert "reprojection rms mm:" in out
    assert "largest gap mm:" in out
```

`cv2` must be imported at the top of `tests/test_detect_pipeline.py`. Importing `_warped_scene` from `tests.test_calib` is allowed; do not copy the scene builder.

- [ ] **Step 5: Run tests, ruff, and sloplint**

Run:

```bash
.venv/Scripts/python -m pytest tests/test_detect_pipeline.py tests/test_markers.py tests/test_quality.py tests/test_synth.py tests/test_calib.py tests/test_coverage.py -v
.venv/Scripts/ruff check .
.venv/Scripts/python tools/sloplint.py
```

Expected: pytest pass, ruff clean, sloplint exit 0.

If the repeatability test fails because the warped ChArUco scene has no dots (largest-gap still prints on a bare pane), that is still success: the line `largest gap mm:` must appear. Do not add fake dots to the calib fixture.

- [ ] **Step 6: Commit**

```bash
git add tests/test_detect_pipeline.py tools/repeatability.py
git commit -m "feat: run coverage on detected markers and report gap spread"
```

---

## Spec coverage

- Spec 4.3 Marker detection: Task 2 (top-hat both polarities, adaptive Gaussian, contour filters, resize to working scale).
- Spec 4.4 Unknown regions: Task 3 (glare = saturated plus low texture; blur = insufficient gradient energy).
- Spec 4.5 consumption of M and U: Task 4 pipeline test. Coverage logic itself is Phase 2 and is not rewritten.
- Spec 6 COOL-relevant ops (resize, adaptive Gaussian, contours): Task 2. No COOL benchmark in this phase.
- Build-plan Phase 3 gate (precision/recall on development captures): synthetic counts in Task 2 stand in until real development captures exist. Record the observed synthetic tp/fp/fn in `PHASE-STATE.md` after the tests run, not as a target written in advance.
- Deferred review item (`repeatability.py` largest-gap spread): Task 4.

## Out of scope

- Real print measurement, phone captures, AWS, COOL, pane quadrilateral UI, fusion, policy, PWA.
- Stripe patterns.
- Tuning thresholds on held-out sessions.
