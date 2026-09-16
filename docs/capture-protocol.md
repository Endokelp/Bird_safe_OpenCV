# Capture protocol

## Print and verify

1. Print `docs/charuco-sheet.pdf` at Actual size (100%). Do not let the print dialog scale to fit.
2. Measure one printed square with a ruler and record it below. Every distance the app reports
   is derived from this value, not from the nominal 30 mm.
3. If the measured value is not 30 mm within 1 mm, the print was scaled. Reprint and remeasure.

**Measured square size: not measured yet**

## Mount the sheet

- Tape the sheet flat against the same glass surface that carries the bird-safe markings, the
  exterior surface. All four edges taped down, no curl.
- Do not tape it to the window frame and do not tape it to the opposite glass surface. Either
  plane sits in front of or behind the markings, which changes the measured scale.

## Capture

- Keep the whole board inside the frame.
- Stand within roughly 45 degrees of square to the glass.
- Fill the frame with the pane.
- Capture from at least three positions per session.

## Record per capture

- Session id
- Pane id
- Date
- Weather
- Time of day
- Background type
- Treatment status: complete, or planted gaps (record each gap's measured position)

## Privacy

Frame the pane, not the surroundings. Avoid people, faces, house numbers, and addresses in the
shot. Captures stay in `captures/`, which is gitignored, and are never committed.
