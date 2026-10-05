"""Tests for the printable calibration sheet PDF."""

import re

from tools.charuco_sheet import SQUARE_MM, SQUARES_X, SQUARES_Y, sheet_footprint_mm, sheet_pdf

CM_RE = re.compile(rb"([\d.]+) 0 0 ([\d.]+) [\d.]+ [\d.]+ cm\n/Im0 Do")
XREF_ENTRY_RE = re.compile(rb"(\d{10}) 00000 n \n")


def test_board_image_is_placed_at_exact_physical_size():
    pdf = sheet_pdf()
    assert pdf[:5] == b"%PDF-"

    match = CM_RE.search(pdf)
    assert match is not None
    width_pt, height_pt = float(match.group(1)), float(match.group(2))
    width_mm = width_pt * 25.4 / 72.0
    height_mm = height_pt * 25.4 / 72.0
    assert abs(width_mm - 210.0) < 0.01
    assert abs(height_mm - 150.0) < 0.01


def test_xref_offsets_point_at_their_object_headers():
    pdf = sheet_pdf()
    start_marker = pdf.rindex(b"startxref")
    xref_offset = int(pdf[start_marker + len("startxref") : pdf.index(b"%%EOF", start_marker)].strip())
    assert pdf[xref_offset : xref_offset + 4] == b"xref"

    offsets = [int(m.group(1)) for m in XREF_ENTRY_RE.finditer(pdf, xref_offset)]
    assert len(offsets) == 6  # objects 1 through 6
    for number, offset in enumerate(offsets, start=1):
        header = f"{number} 0 obj".encode("latin-1")
        assert pdf[offset : offset + len(header)] == header


def test_footprint_is_a_us_letter_landscape_page_that_contains_the_board():
    x0, y0, x1, y1 = sheet_footprint_mm()
    assert abs((x1 - x0) - 279.4) < 0.1
    assert abs((y1 - y0) - 215.9) < 0.1
    assert x0 <= 0 and y0 <= 0
    assert x1 >= SQUARES_X * SQUARE_MM and y1 >= SQUARES_Y * SQUARE_MM
