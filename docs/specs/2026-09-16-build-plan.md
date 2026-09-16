# Build plan: phases, gates, and review protocol

Date: 2026-09-16. Spec: `2026-09-16-coverage-checker-spec.md`. Rules: `AGENTS.md`.

Deadline: 2026-10-26, 11:45 p.m. PDT (the earlier of the two published times). 40 days from today.
The user submits. Agents own software, AWS, OpenCV, evaluation, docs, and the demo build.

## Verified facts this plan rests on

| Fact | Status | Source |
|---|---|---|
| OpenCV 5.0.0 released 2026-06-06 | Verified | opencv.org release page |
| `opencv-python==5.0.0.93` is a PyPI wheel | Verified | `pip index versions opencv-python` run 2026-09-16, local machine has 4.13.0.92 |
| ArUco and ChArUco are in `objdetect`, not contrib | Verified as of OpenCV 4.7 | OpenCV module layout |
| COOL is an AWS Marketplace AMI for Graviton3 and Graviton4 | Verified | AWS Marketplace listing |
| COOL software fee about $0.01/hr at `c8g.large`, 7-day free trial | Verified | AWS Marketplace pricing page |
| COOL accelerates resize, adaptive Gaussian threshold, contour detection | Verified | OpenCV COOL announcement |
| ABC 2x2 rule: markings no more than 2 inches apart, exterior surface | Verified | American Bird Conservancy |
| AWS free plan: $100 credits, up to $200, 6 months, new accounts | Verified | AWS announcement 2025-07-15 |
| Compute grant path | Closed, deadline passed | User, 2026-09-16 |

Anything not in this table gets verified before a phase depends on it.

## Sequencing logic

Two tracks run in parallel from day one, because decal shipping is the longest lead item and no
code should wait on it.

- Physical track: order decals, print the calibration board, apply a treated pane with deliberate
  measured gaps, capture sessions. The user does the physical work. Agents produce the printable
  target, the capture protocol, and the ground-truth record format.
- Software track: build and validate against synthetic panes where the correct answer is known
  exactly, then swap in real captures the moment they exist.

Phase 2 is a hard gate. If metric measurement is not repeatable on real captures, the claim gets
narrowed or the capture protocol gets tightened before any interface work starts. `AGENTS.md`
requires this and the plan does not route around it.

## Phases

### Phase 0: Foundations and guardrails (Sep 16 to Sep 17)

Repo, dependency pins, CI, and the slop gate. Detailed tasks in
`docs/plans/2026-09-16-phase-0-2-vision-core.md`.

Gate: CI green. `git ls-files` contains none of `WIN-AND-PLACE-CLAUDE.txt`, `CONTEST-DOSSIER.md`,
`MASTER-BRIEF.md`. `python -c "import cv2; print(cv2.__version__)"` prints 5.0.0.

### Phase 1: Capture fixture and synthetic ground truth (Sep 16 to Sep 21, parallel)

User action on day one: order a roll of bird-safe dot decals or window dot tape. Without this
nothing in Phase 3 onward has real evidence behind it.

Agent deliverables: a printable ChArUco board PDF with the square size stated on the sheet, a
written capture protocol, a ground-truth layout record format, and a synthetic pane generator that
renders a rectified pane with a known dot layout, known planted gaps, and optional synthetic glare.

Gate: the synthetic generator emits a pane whose violating regions are known by construction, and
that ground truth round-trips through the record format.

### Phase 2: Metric core (Sep 18 to Sep 23) HARD GATE

`calib.py` and `coverage.py`. The two modules the entry lives or dies on.

Gate, in two parts:
1. Synthetic: on generated panes, the coverage check finds every planted violation, reports zero
   violations on a correctly treated pane, and the largest-gap size error stays under 1 mm at a
   stated scale.
2. Real: ten captures of the same physical pane from different positions agree on millimetres per
   pixel and on largest-gap size within a tolerance recorded in `PHASE-STATE.md`.

If part 2 fails, tighten the capture protocol or narrow the supported angle range and repeat. Do
not proceed to interface work on an unrepeatable measurement.

### Phase 3: Detection and quality masks (Sep 22 to Sep 28)

`markers.py` and `quality.py`, tuned on development captures only. Held-out sessions stay sealed.

Gate: precision and recall reported on development captures with the numbers written down as
observed, not as targets. Glare and blur masks visibly track the regions a human marks as
unreadable on at least one hard capture.

### Phase 4: Fusion and the decision policy (Sep 27 to Oct 2)

`fusion.py` and `policy.py`, plus the trace record format.

Gate: unknown area fraction measurably falls as frames are added on a real multi-frame session, and
a recorded trace replays to the same decisions from its logged metrics.

### Phase 5: Evaluation (Oct 2 to Oct 8)

Held-out harness, the full metrics table from spec section 8, the single-frame baseline, and the
failure case catalogue.

Gate: the metrics table is produced from held-out captures that were never used for tuning, and the
baseline comparison is in it, including any metric where the pipeline does not beat the baseline.

### Phase 6: AWS service and mobile app (Oct 5 to Oct 14)

`api.py`, the PWA, S3 hosting, the Graviton deployment, CloudWatch, and the capture retention rule.

Gate: a phone on cellular, not on the local network, completes a scan through the public URL and
sees the overlay. Observed, with a screen recording kept.

### Phase 7: COOL benchmark (Oct 12 to Oct 16)

The same pipeline built against COOL and against stock OpenCV 5 on the same Graviton instance type,
measured over the same capture set.

Gate: a results table with instance type, both build identities, per-stage latency, end-to-end
latency, throughput, and cost. Whatever the number is, it gets published as measured. Run inside
the 7-day free trial.

### Phase 8: Field test and polish (Oct 15 to Oct 20)

On-phone testing in at least three lighting conditions, plus the deliberate failure paths: heavy
glare, a calibration board that cannot be found, a capture too far away to resolve dots, and a
narrow screen.

Gate: each failure path produces a specific, actionable message rather than a generic error, and
each is captured in a screenshot or recording.

### Phase 9: Documentation, video, submission (Oct 19 to Oct 25)

Report, architecture diagram, README with pinned build instructions, responsible-use statement,
failure cases, the AWS run, and a video within the five-minute limit. Submission dry run, then the
user submits.

Gate: submitted before Oct 26, 11:45 p.m. PDT, with the endpoint reachable.

Buffer: one day. It is thin. Phases 3 through 5 are the most likely to slip, and the release valve
is Phase 8 polish, not Phase 5 evaluation. Evidence outranks polish on this rubric.

## Rubric traceability

Every phase exists to produce evidence for a scored criterion. A deliverable with no cell here does
not get built.

| Criterion | Weight | Produced by |
|---|---|---|
| Technical execution | 30% | Phases 2, 3, 4, and the Phase 5 numbers |
| Innovation | 20% | Phase 2 exact coverage check, Phase 4 decision loop |
| Real-world impact | 20% | Phase 1 real material, Phase 5 held-out evaluation and failure cases |
| User experience | 10% | Phases 6 and 8 |
| Docs and presentation | 10% | Phase 9 |
| Cloud, reproducibility, responsible operation | 10% | Phases 0, 6, 7 |
| Agentic Vision award | separate | Phase 4 trace |
| Best Use of COOL award | separate | Phase 7 measured table |

## Subagent assignment

One writer per file set, per `AGENTS.md` and the CodingBrain contract. Modules in the same row may
run in parallel because they share no files. Interfaces are fixed in the task plan before any
parallel dispatch, so no subagent invents a signature another subagent has to guess.

| Wave | Parallel subagents | Owned paths |
|---|---|---|
| 0 | one | repo root, `pyproject.toml`, `.github/`, `.gitignore`, `tools/sloplint.py` |
| 1 | two | A: `tools/charuco_sheet.py`, `docs/capture-protocol.md`. B: `birdsafe/synth.py` |
| 2 | two | A: `birdsafe/calib.py`. B: `birdsafe/coverage.py` |
| 3 | two | A: `birdsafe/markers.py`. B: `birdsafe/quality.py` |
| 4 | two | A: `birdsafe/fusion.py`. B: `birdsafe/policy.py` |
| 5 | one | `eval/` |
| 6 | two | A: `birdsafe/api.py`, `infra/`. B: `web/` |
| 7 | one | `bench/` |
| 8 | one | `web/` polish, owned solo to avoid a second writer |
| 9 | two | A: `README.md`, `docs/report.md`. B: video script and assets |

Rules for every dispatch:

- The subagent receives the spec section it implements, its owned paths, the exact interface it
  consumes and produces, and its acceptance check. Nothing else.
- A subagent never edits a file outside its owned paths. If it needs a change elsewhere, it reports
  the need and the orchestrator routes it.
- Every subagent leaves one runnable check behind that fails if its logic breaks.

## Review protocol

Each phase gate runs three independent reviews before the phase is marked done. They look for
different things and none of them substitutes for running the code.

1. Codex: logic and correctness against the real diff. Off-by-one in the erosion kernel size, unit
   confusion between millimetres and pixels, mask polarity inversions, silent exception swallowing.
2. Cursor with Grok 4.6 at extra-high: architecture and edge cases against the spec, with the spec
   section and the diff supplied.
3. Grok bot slop detector: repository, code comments, and prose. Also the `tools/sloplint.py` gate,
   which is the mechanical half and runs in CI on every push.

Reviews run against a named revision. Findings and their verification go into `PHASE-STATE.md`
against that revision. Agreement between reviewers is not proof; the gate is the observed check.

## Anti-slop gate

Mechanical, in CI, failing the build:

- No em dash anywhere in tracked files.
- No banned vocabulary: delve, leverage, seamless, revolutionize, game-changing, cutting-edge,
  unlock, elevate, robust solution, in today's fast-paced world.
- No "not just X, but Y" construction.
- No `console.log` or bare `print` debugging in committed source.
- No `TODO` without an owner and a reason.
- Strategy documents absent from `git ls-files`.

Judgement, at each review:

- Comments state a non-obvious reason or they get deleted.
- No abstraction with one implementation. No config for a value that never changes.
- No number in any document without an observed run behind it.
- No capability claim the demo cannot show.

## What the user does

Everything else is on the agents.

1. Today: order bird-safe dot decals or window dot tape.
2. Confirm whether an AWS account already exists, because a new account gets the credits that make
   this free.
3. Print the ChArUco sheet on plain paper when Phase 1 delivers it, without scaling in the print
   dialog.
4. Apply the decals to a window or glass door, leaving deliberate gaps at measured positions.
5. Capture the sessions the protocol asks for.
6. Record the video and submit.
