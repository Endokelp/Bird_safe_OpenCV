# Capture protocol

## Print and verify

1. Print `docs/charuco-sheet.pdf` at Actual size (100%). Do not let the print dialog scale to fit.
2. Measure the full width of the printed board, across all seven squares, with a ruler. It must
   read 210 mm within 1 mm. Measuring the whole width rather than one square keeps a 1 mm ruler
   reading error to about half a percent of scale.
3. If it does not, the print was scaled. Reprint and remeasure. The app assumes the nominal
   board size, so a scaled print makes every reported distance wrong by the same percentage.

**Measured board width: 210 mm, reported by the user on 2026-10-20.**

## Mount the sheet

- Tape the sheet flat against the same glass surface that carries the bird-safe markings, the
  exterior surface. All four edges taped down, no curl.
- Align the board's edges with the pane's edges, long edge parallel to the sill. The 2x2 check
  tests horizontal and vertical gaps along the board's axes, so a tilted board tests the wrong
  directions and can miss a gap.
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
