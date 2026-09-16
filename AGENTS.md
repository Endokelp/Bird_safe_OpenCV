# OpenCV AI Competition 2026: workspace rules

Canonical project instructions for Claude Code, Codex, Grok Bot, Cursor, and other agents working
in this folder. Read `PHASE-STATE.md` for current status. The user's current request and the
global CodingBrain contract still apply. Do not import goals or product decisions from SpaceApp.

## Mission and settled decision

Build one entry for the OpenCV AI Competition 2026: a phone-based checker for the installation
coverage of visible bird-safe window markings. A person scans a treated pane, sees where dots
or stripes are missing or too far apart, repairs the marked area, and scans it again. The simple
pitch is: **find the gaps in a bird-safe window treatment before the job is signed off.**

The user selected this idea on September 16, 2026 because real footage and an honest demo are
obtainable. Do not reopen the three-idea contest, change the target domain, or add a second
product without the user's instruction. Aim for first place overall and compete for Agentic Vision
and Best Use of COOL when the implementation actually meets those rubrics. No award is promised.

Official competition: https://opencv26.devpost.com/ and https://opencv26.devpost.com/rules .
Final deadline in the page body is October 26, 2026 at 11:59 p.m. Pacific; the Devpost banner
says 11:45 p.m. PDT. Plan to finish and submit before the earlier time. The user handles the
actual submission.

## The rubric is the build specification

Technical execution 30%; innovation 20%; real-world impact 20%; user experience 10%; docs and
presentation 10%; cloud delivery, reproducibility, and responsible operation 10%. A feature
earns its place by improving the user's scan-to-repair task or producing evidence for one of
these criteria. Do not add features for a longer feature list.

The required core is substantive OpenCV 5 image or video analysis plus a meaningful AWS
component. Treat the special awards as separate gates:

- Agentic Vision: an OpenCV result changes a subsequent capture request, analysis step,
  repair instruction, or human approval. Preserve a trace showing the change. A chatbot
  describing a fixed result or an AI coding agent writing the app does not qualify.
- COOL: the claimed core workload must execute through COOL on AWS Graviton or a documented
  Arm path. Record its version, workload, baseline, latency or throughput, cost, and limitations.
  Do not claim a COOL speedup without measured results. COOL is optional for the overall prize.

## Product boundary

- First working slice: visible dot treatments on one pane, with a known physical scale. Detect
  pane corners, rectify perspective, find markings, measure uncovered gaps, and overlay exact
  repair locations. Add stripes or other patterns only after the dot workflow passes field tests.
- Use multiple frames when they improve glare handling or coverage. Mark obscured regions
  **unknown** and ask for a targeted rescan. Never infer a physical distance without a valid
  scale reference or calibration.
- A human confirms that markings are on the exterior surface and reviews the final finding.
  The app may report measurements against cited guidance. It must not certify a window as
  bird-safe, code-compliant, or proven to prevent collisions.
- Phone and laptop are enough for the core demo. Ask the user only for small physical setup,
  short real captures, presentation or video help, and final submission. Agents own the software,
  OpenCV, AWS, evaluation, documentation, and build steps.
- No bird detector, collision predictor, generic property-risk dashboard, AI chat, maps, social
  features, or accounts unless a measured user need makes one essential.

## Evidence gates

1. Feasibility first: test real phone footage of a correctly marked sample and samples with
   deliberately missing markings. Keep a ruler or known marker in view. Record glare, angle,
   and lighting failures. If physical gap measurements are unreliable, improve capture guidance
   or narrow the claim before building a large interface.
2. Evaluation: use a held-out set of real panes or realistic window samples with labeled
   gaps and known scale. Report precision, recall, gap-size error, false alarms, uncertain area,
   and performance by lighting/background. Compare with a simple single-frame baseline. Keep
   training, tuning, and test captures separated by pane or capture session.
3. Submission: demonstrate one complete scan, a detected gap, the requested recapture or
   repair, and a rescan. Include a judge-accessible endpoint or arranged live demo, pinned build
   instructions, architecture, real failure cases, an AWS run, and a video within the official
   five-minute limit. No staged result may be described as a real installation outcome.

Never claim a test, phone run, AWS deployment, COOL benchmark, or user result passed without
observing it. Cite the actual artifact and revision. If a check fails, record the failure and
the next fix in `PHASE-STATE.md`.

## How agents work here

- Start with `PHASE-STATE.md` and the active checkpoint, then inspect relevant files and Git
  changes. This folder had no Git repository at idea lock; verify again before using Git.
- Work only in `C:\Users\venni\OpenCV`. Read other projects only when the user asks for a
  reference; do not edit them. Keep private strategy and competitor notes out of a future
  public submission repository.
- Reuse current code and standard tools before adding dependencies or abstractions. Verify
  OpenCV 5, COOL, and AWS APIs against installed versions or official docs before coding to
  them. Do not add paid services or purchases without a clear need and authorization.
- Capture only what the analysis needs. Crop to panes where possible, avoid identifiable
  people or addresses, get consent for shared footage, and define retention before uploading
  captures to AWS or placing them in a public dataset.
- One writer owns a file set. Delegate only bounded independent work with paths and checks.
  Review consequential changes against the real diff. Save a checkpoint after each milestone
  and before handoff: goal, owner, changed files or revision, observed checks, unresolved risk,
  and exact next action.
- Current user decisions stay settled until the user changes them. Challenge a weak premise
  with brief evidence, then continue the agreed route unless it is infeasible.

## No slop in code, product, or submission

- Write plain, specific language. No hype, invented novelty, invented impact, fake confidence
  percentages, or "AI-powered" as a substitute for explaining the vision step.
- Avoid em dashes, decorative emoji, canned openings, and phrases such as "delve," "leverage," "seamless," "revolutionize," "game-changing," and "in today's fast-paced world." (sloplint: allow)
- Do not use the formula "not just X, but Y." Show the observed behavior and number instead. (sloplint: allow)
- Comments explain a non-obvious reason. No narrated steps, dead code, banner comments, or TODO without an owner and a reason. (sloplint: allow)
  Names should describe the pane, marker, gap, calibration, or capture they represent.
- Design one clear scan-to-result path with visible uncertainty and a repair location. Test it
  on an actual phone, including poor light, glare, a failed calibration, keyboard access where
  applicable, and narrow screens. A screenshot alone is not a usability test.
