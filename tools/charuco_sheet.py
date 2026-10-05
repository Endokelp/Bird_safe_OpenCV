"""Renders the printable calibration board. Physical square size must match birdsafe.calib."""

import zlib

import cv2

SQUARE_MM = 30.0
MARKER_MM = 22.0
SQUARES_X, SQUARES_Y = 7, 5
DICT = cv2.aruco.DICT_4X4_50

PAGE_W_PT, PAGE_H_PT = 792.0, 612.0  # US Letter landscape
MM_TO_PT = 72.0 / 25.4
DPI = 300.0
TEXT_SIZE_PT = 10.0
BOARD_W_PT = SQUARES_X * SQUARE_MM * MM_TO_PT
BOARD_H_PT = SQUARES_Y * SQUARE_MM * MM_TO_PT


def birdsafe_board() -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_MM / 1000.0,
        MARKER_MM / 1000.0,
        cv2.aruco.getPredefinedDictionary(DICT),
    )


def _pdf_object(number: int, body: bytes) -> bytes:
    return f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"


def _layout_pt() -> tuple[float, float, float]:
    """Returns (board_x, board_y, text_baseline) in PDF points, origin at the page's bottom-left."""
    board_x = (PAGE_W_PT - BOARD_W_PT) / 2.0

    gap_pt = 16.0
    baseline_margin_pt = 8.0
    block_h = BOARD_H_PT + gap_pt + TEXT_SIZE_PT + baseline_margin_pt
    block_bottom = (PAGE_H_PT - block_h) / 2.0
    text_baseline = block_bottom + baseline_margin_pt
    board_y = text_baseline + TEXT_SIZE_PT + gap_pt
    return board_x, board_y, text_baseline


def sheet_footprint_mm() -> tuple[float, float, float, float]:
    """Returns (x0, y0, x1, y1) of the printed page in the board frame.

    The board frame has its origin at the board's top-left square corner and y pointing down.
    """
    board_x, board_y, _text_baseline = _layout_pt()
    board_top = board_y + BOARD_H_PT
    return (
        -board_x / MM_TO_PT,
        -(PAGE_H_PT - board_top) / MM_TO_PT,
        (PAGE_W_PT - board_x) / MM_TO_PT,
        board_top / MM_TO_PT,
    )


def sheet_pdf() -> bytes:
    """Builds a one-page PDF with the board printed at exact physical size.

    A PNG carries no physical size, so a print dialog scales it to fit the page. A PDF states
    the image's placement in points, which a compliant reader and printer honour at 100%.
    """
    px_per_mm = DPI / 25.4
    img_w = round(SQUARES_X * SQUARE_MM * px_per_mm)
    img_h = round(SQUARES_Y * SQUARE_MM * px_per_mm)
    image_bytes = zlib.compress(birdsafe_board().generateImage((img_w, img_h)).tobytes())

    board_x, board_y, text_baseline = _layout_pt()

    text = (
        f"{SQUARES_X} x {SQUARES_Y} squares, {SQUARE_MM:.0f} mm squares, "
        f"{MARKER_MM:.0f} mm markers, DICT_4X4_50. "
        "Print at Actual size \\(100%\\). "
        f"Board width must measure {SQUARES_X * SQUARE_MM:.0f} mm."
    )

    content = (
        "q\n"
        f"{BOARD_W_PT:.6f} 0 0 {BOARD_H_PT:.6f} {board_x:.6f} {board_y:.6f} cm\n"
        "/Im0 Do\n"
        "Q\n"
        "BT\n"
        f"/F1 {TEXT_SIZE_PT:.2f} Tf\n"
        f"{board_x:.6f} {text_baseline:.6f} Td\n"
        f"({text}) Tj\n"
        "ET\n"
    ).encode("latin-1")

    obj1 = _pdf_object(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    obj2 = _pdf_object(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    obj3 = _pdf_object(
        3,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 792 612] "
        b"/Resources << /XObject << /Im0 4 0 R >> /Font << /F1 5 0 R >> >> "
        b"/Contents 6 0 R >>",
    )
    obj4 = _pdf_object(
        4,
        f"<< /Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} "
        f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
        f"/Length {len(image_bytes)} >>\nstream\n".encode("latin-1")
        + image_bytes
        + b"\nendstream",
    )
    obj5 = _pdf_object(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    obj6 = _pdf_object(
        6,
        f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream",
    )

    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in (obj1, obj2, obj3, obj4, obj5, obj6):
        offsets.append(len(buf))
        buf += obj

    xref_offset = len(buf)
    count = len(offsets) + 1  # object 0 is the free-list head
    buf += f"xref\n0 {count}\n".encode("latin-1")
    buf += b"0000000000 65535 f \n"
    for offset in offsets:
        buf += f"{offset:010d} 00000 n \n".encode("latin-1")
    buf += (
        f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
    ).encode("latin-1")

    return bytes(buf)


def main() -> None:
    with open("docs/charuco-sheet.pdf", "wb") as fh:
        fh.write(sheet_pdf())


if __name__ == "__main__":
    main()
