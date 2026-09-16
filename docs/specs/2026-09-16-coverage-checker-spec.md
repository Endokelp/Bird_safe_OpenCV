# Build spec: bird-safe window treatment coverage checker

Date: 2026-09-16. Status: proposed, awaiting approval.
Canonical rules: `AGENTS.md`. Current state: `PHASE-STATE.md`. Phase plan: `2026-09-16-build-plan.md`.

## 1. Context

Bird-safe window treatments work only when the markings are spaced tightly enough. The American
Bird Conservancy states the 2x2 rule: markings no more than 2 inches apart horizontally and
vertically, applied to the exterior surface of the glass. A 2023 William and Mary study found that
films applied to the interior of double-paned glass did not reduce collisions, because the exterior
reflection hides them.

The failure mode in practice is not the product, it is the installation. Decal rolls drift out of
alignment, an applicator skips a band near a mullion, dots fall off after a season, and nobody
measures the result. The person who applied the treatment has no way to check spacing across a
whole pane other than holding up a ruler.

This entry builds a phone-based checker that answers one closed question against a published
standard: does any untreated region on this pane violate the 2x2 rule, and where exactly is it.

## 2. Scope

In scope for the first working slice:

- Visible dot treatments on a single pane, with a printed calibration target in frame.
- Metric rectification, marker detection, gap measurement in millimetres, repair overlay.
- Multi-frame capture with an explicit decision policy that requests targeted recaptures.
- Held-out evaluation on real captures, with a single-frame baseline for comparison.
- AWS deployment with a judge-reachable endpoint, and a measured COOL benchmark on Graviton.

Out of scope, and stated as such in the submission:

- Stripe, dot-and-line, acid-etch, and UV-reflective patterns. Dots first, the rest only if the dot
  workflow passes field tests with time to spare.
- Any bird detector, collision predictor, risk dashboard, chat interface, map, account system, or
  property database.
- Certifying a window as bird-safe or code-compliant. The app reports measurements against cited
  guidance. A human confirms exterior placement and reviews the finding.

## 3. Users and the task

Primary user: the person who just applied the treatment, standing at the window, holding a phone.
Their task is to scan, see whether it passes, fix the marked spots, rescan, and get a clean result.

Secondary user: a building manager or volunteer coordinator checking work that someone else did,
who needs a record with measurements rather than an opinion.

Success for the user is a repair location they can walk to and patch, not a score.

## 4. Core method

All geometry happens on a rectified metric plane where one pixel is a known number of millimetres.
Nothing downstream reasons in pixels.

### 4.1 Scale and rectification

A printed ChArUco board is taped to the glass or held flat against it, inside the capture frame.
OpenCV 5 `objdetect` detects the board, which yields both the homography to a fronto-parallel view
and the millimetres-per-pixel scale from the known square size. One printed sheet solves scale,
pose, and rectification together.

This removes the largest correctness risk in the project. The app never infers a physical distance
from an assumed pane size or an assumed camera. If the board is not found, the capture is rejected
and the user is told why. There is no silent fallback to a guessed scale.

Fallback path, built only if field tests demand it: the user taps four pane corners and enters one
real edge length. Lower accuracy, and labelled as such in the result.

### 4.2 Pane region

The user drags a quadrilateral over the glass on the rectified preview. Two corner drags define the
region under test. Automatic glass segmentation is not built. It would add a model, a failure mode,
and a dependency to replace an interaction that takes three seconds.

### 4.3 Marker detection

Dots are small high-contrast blobs of roughly known physical diameter, and the treatment may be
light dots on a dark background or dark dots on a light background. The detection chain is:

1. Grayscale, then resize to a fixed working scale.
2. White top-hat and black top-hat with a structuring element sized from the expected dot diameter
   in millimetres. This isolates small blobs of either polarity and removes the background, whether
   that background is sky, foliage, or a lit interior.
3. Adaptive Gaussian threshold on each top-hat response.
4. `findContours`, then filter by area, circularity, and physical diameter in millimetres.

Every one of resize, adaptive Gaussian threshold, and contour detection is on the published list of
operations COOL accelerates on Graviton. The COOL benchmark therefore measures the real pipeline,
not a synthetic microbenchmark chosen to look good.

### 4.4 Unknown regions

A pixel is unknown when the analysis cannot see the glass surface. Two sources:

- Glare: near-saturated value with low local texture variance.
- Occlusion or blur: local gradient energy below the level needed to resolve a dot of the expected
  diameter, measured against the current millimetres-per-pixel.

Unknown area is never counted as covered and never counted as a gap. It is reported as its own
number, and it is the main trigger for the decision policy in section 5.

### 4.5 Gap measurement

This is the part a judge should be able to check in one reading.

Let `M` be the marker mask and `U` the unknown mask on the rectified metric plane, both clipped to
the pane quadrilateral. Empty space is `E = pane AND NOT M`.

The 2x2 rule is violated wherever an axis-aligned 2 inch by 2 inch square fits inside the pane
containing no marking. That is exactly a morphological erosion. Erode `E` with a square structuring
element of side `50.8 mm / mm_per_px`, with the border treated as non-empty so that only squares
fully inside the pane are tested. Every surviving pixel is the centre of an all-empty 2 inch square,
which is a violation. Dilating the survivors back by the same element draws the violating region.

Two masks are produced, and both are reported:

- Confirmed violation: the square is all-empty and fully observed, `E AND NOT U`.
- Possible violation: the square fits in `pane AND (NOT M OR U)` but not in `E AND NOT U`. A marker
  detected inside unknown area does not count as coverage. Not claimed as a failure. It becomes a
  recapture request.

The square side in pixels is the smallest whole number strictly greater than `50.8 mm / mm_per_px`,
so a gap of exactly 2 inches passes.

Headline number: the side of the largest all-empty axis-aligned square in `E AND NOT U`, reported in
inches next to the 2 inch threshold. It is the same shape the verdict tests, so the number exceeds
2 inches exactly when a confirmed violation exists, and unknown area never counts as a gap. A
chessboard distance transform with a zero border gives it to within one pixel, and one erosion
settles that pixel. An earlier draft used the largest empty circle; on a compliant 45 mm dot grid
that read 57.4 mm while the square test passed, so the number contradicted the verdict.

Round dots let a square slip diagonally between them, so on a regular grid of 6 mm dots the check
flags a centre spacing of 56 mm even though the horizontal edge-to-edge gap is about 49.5 mm. This
follows from reading the rule as "no empty 2 inch square", and the report states that reading.

The check is axis-aligned because the rule is stated in horizontal and vertical terms. A region that
is wide only on the diagonal is not reported, and that interpretation is stated in the report rather
than left implicit.

Connected components of the confirmed violation mask become repair targets, each with a centroid in
millimetres from a named pane corner, which is what the overlay draws and what the report lists.

The whole check is three erosions, one distance transform, and one connected-components call. It is
exact rather than learned, it runs in milliseconds, and its failure modes are all upstream in
detection rather than hidden inside a model.

### 4.6 Multi-frame fusion

Captures from several positions are rectified onto the same metric plane through the same ChArUco
board, so they share one coordinate system with no registration step.

- Markings fuse by union. A dot seen in any frame is a dot.
- Unknown fuses by intersection. A region is unknown only if no frame resolved it.

This turns extra captures into a measurable reduction in unknown area, which is the number the
evaluation reports against frame count.

## 5. The decision policy

Vision output changes what happens next, through an explicit rule table rather than a language
model. A deterministic policy is easier to defend, cheaper to run, and reproducible from the trace.
A language model describing a finished result would not qualify for the Agentic Vision award and is
not used.

After every capture the analyser emits metrics, and the policy selects one action:

| Observed metric | Condition | Action requested |
|---|---|---|
| ChArUco corner reprojection RMS | above threshold | Recapture, board not flat or motion blur |
| Angle from board normal | too oblique | Recapture from a squarer angle |
| Millimetres per pixel | too coarse to resolve a dot | Move closer, resolution insufficient |
| Unknown area fraction | above threshold | Recapture aimed at the unknown centroid, from an offset position that moves the glare |
| Possible violation present, confirmed absent | any | Targeted recapture of that region only |
| Confirmed violation present | any | Emit repair plan with millimetre coordinates |
| Rescan after repair | prior target region | Verify that region now passes, log before and after |
| All clear | no violation, unknown under threshold | Report pass, request human confirmation of exterior placement |

Every decision writes a trace record: input metrics, rule fired, action requested, and the metric
change measured after the resulting capture. That trace is the Agentic Vision evidence, and it is
the same artifact used to evaluate whether the loop actually helps.

The loop is capped. After a fixed number of recapture requests without improvement, the app stops
asking and reports what it can see plus what it could not, rather than looping on a user.

## 6. Architecture

```
phone browser (PWA on S3)
  capture frame, draw overlay, show requested action
        |  HTTPS, JPEG frame + session id
        v
analysis service (FastAPI, AWS Graviton, OpenCV 5 built against COOL)
  rectify -> detect -> unknown mask -> fuse -> coverage check -> policy -> trace
        |
        v
S3 (captures and traces, lifecycle expiry)   CloudWatch (latency, error rate, decisions)
```

Module boundaries, one writer each:

| Module | Responsibility | Depends on |
|---|---|---|
| `birdsafe/calib.py` | ChArUco detection, homography, mm-per-pixel, reprojection error | OpenCV |
| `birdsafe/markers.py` | Top-hat, threshold, contours, physical-size filter | OpenCV, calib output |
| `birdsafe/quality.py` | Glare and blur masks, unknown mask | OpenCV, calib output |
| `birdsafe/coverage.py` | Erosion check, distance transform, repair targets | masks only, no image I/O |
| `birdsafe/fusion.py` | Multi-frame union and intersection on the metric plane | masks only |
| `birdsafe/policy.py` | Rule table, action selection, trace records | metrics only, no OpenCV |
| `birdsafe/api.py` | FastAPI routes, session state, S3 writes | all of the above |
| `web/` | PWA capture, overlay, guidance | HTTP contract only |
| `eval/` | Held-out harness, metrics, baseline comparison | library, not the API |
| `bench/` | COOL versus stock OpenCV 5 benchmark on Graviton | library, not the API |

`coverage.py` and `policy.py` take arrays and numbers, never files or network. They are the two
modules whose correctness decides the entry, and both are testable with no camera and no cloud.

## 7. AWS and cost

Target: zero out-of-pocket using the AWS free plan for new accounts, which currently grants $100 in
credits, up to $200 after onboarding activities, over six months. Hard ceiling if credits are
unavailable: $30.

- S3 static hosting for the PWA, plus a captures bucket with a lifecycle expiry rule.
- One Graviton instance running the analysis service. COOL is an AWS Marketplace AMI for the
  Graviton3 and Graviton4 families with a software fee around $0.01 per hour at `c8g.large` on top
  of EC2, and a 7-day free trial.
- CloudWatch metrics and logs for latency, error rate, and policy decisions.

Cost control, in order:

1. Run the COOL benchmark inside the 7-day free trial window.
2. Keep the instance stopped outside development sessions and demo windows.
3. Before submission, confirm the cost of leaving the endpoint running through the judging period.
   If that exceeds the ceiling, keep the endpoint on a smaller Arm instance with a stock OpenCV 5
   build, and state plainly in the report which path served which measurement.

No claim of a COOL speedup is made without a measured result on the same instance type against a
stock OpenCV 5 build of the same pipeline.

## 8. Evaluation

Ground truth comes from the fact that we apply the decals ourselves and therefore know the intended
layout, plus deliberately removed dots at measured positions.

Split by capture session and by pane, never by frame. Tuning captures and held-out captures never
share a pane or a session.

Reported metrics:

- Marker detection precision and recall.
- Largest-gap size error in millimetres against the known layout.
- Repair target localisation error in millimetres.
- False alarm rate, meaning confirmed violations reported on a pane that is correctly treated.
- Unknown area fraction, against number of frames.
- All of the above broken down by lighting condition and background type.

Baseline for comparison: single frame, fixed global threshold, no decision policy, no fusion. The
report shows the delta the pipeline actually buys, including any case where it buys nothing.

Failure cases are documented rather than hidden. Expected ones: strong direct reflection of a bright
sky, very oblique angles, dots below the resolvable size at the capture distance, low contrast
between marking and background, and wet or dirty glass.

## 9. Honesty constraints

These are build requirements, not style preferences.

- No accuracy, latency, or speedup number appears anywhere without an observed run behind it.
- Synthetic and staged captures are labelled as synthetic and staged in every artifact.
- The app never certifies a window as bird-safe, code-compliant, or collision-proof.
- Exterior placement cannot be verified from these images and is always a human confirmation step.
- No em dashes, no hype vocabulary, no invented novelty, in code, docs, repo, or video.

## 10. Repository hygiene

The public repository is `Endokelp/Bird_safe_OpenCV`, currently private with no commits, so nothing
has been published yet.

`WIN-AND-PLACE-CLAUDE.txt`, `CONTEST-DOSSIER.md`, and `MASTER-BRIEF.md` contain competition
strategy, judge names, and prize-stacking notes. They must be excluded by `.gitignore` before the
first commit. Once a file enters Git history it stays there, so this happens in Phase 0 and not
later.

## 11. Open risks

| Risk | Handling |
|---|---|
| Decal shipping time delays real captures | Order on day one, and build and validate on synthetic panes with exact known answers in parallel |
| Glare makes a large fraction of real panes unknown | This is what the decision policy and fusion exist to reduce, and the reduction is a reported number rather than an assumption |
| COOL speedup on this pipeline is small | Report the measured number whatever it is. A small honest number scores better than an unsupported large one |
| Endpoint cost through judging | Section 7, item 3 |
| Entrant eligibility and proposal requirements | Verify against the official rules before submission, tracked in `PHASE-STATE.md` |
