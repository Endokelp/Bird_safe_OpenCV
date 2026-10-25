import base64
import json
import uuid

import cv2
import numpy as np
import pytest

from birdsafe import api
from birdsafe.coverage import RULE_MM
from tools.charuco_sheet import SQUARE_MM, SQUARES_X, SQUARES_Y, birdsafe_board

MM_PER_PX = 0.5
DOT_MM = 19.05
# Board top-left in the scene, and a 4 x 4 dot grid at 40 mm to the right of the sheet, the
# session01 layout.
BOARD_AT_PX = (60, 160)
GRID_FROM_BOARD_MM = (270.0, 0.0)


def _scene(omit=(), spacing_mm=40.0):
    rng = np.random.default_rng(7)
    board_w = round(SQUARES_X * SQUARE_MM / MM_PER_PX)
    board_h = round(SQUARES_Y * SQUARE_MM / MM_PER_PX)
    bx, by = BOARD_AT_PX
    grid_px = 3 * spacing_mm / MM_PER_PX
    width = max(1100, round(bx + GRID_FROM_BOARD_MM[0] / MM_PER_PX + grid_px + 80))
    height = max(640, round(by + grid_px + 80))
    # Textured glass: the blur test needs some background detail.
    scene = np.clip(rng.normal(150, 8, (height, width)), 0, 255).astype(np.uint8)
    scene[by - 40 : by + board_h + 40, bx - 20 : bx + board_w + 20] = 255
    scene[by : by + board_h, bx : bx + board_w] = birdsafe_board().generateImage((board_w, board_h))
    for row in range(4):
        for col in range(4):
            if (row, col) in omit:
                continue
            x = bx + (GRID_FROM_BOARD_MM[0] + spacing_mm * col) / MM_PER_PX
            y = by + (GRID_FROM_BOARD_MM[1] + spacing_mm * row) / MM_PER_PX
            cv2.circle(scene, (round(x), round(y)), round(DOT_MM / 2 / MM_PER_PX), 60, -1, cv2.LINE_AA)
    return cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)


def _jpeg(image):
    return base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]).decode()


def _post(body):
    event = {"requestContext": {"http": {"method": "POST"}}, "rawPath": "/scan", "body": json.dumps(body)}
    response = api.handler(event, None)
    return response["statusCode"], json.loads(response["body"])


@pytest.fixture(autouse=True)
def local_sessions(tmp_path, monkeypatch):
    monkeypatch.delenv("SESSION_BUCKET", raising=False)
    monkeypatch.setenv("SESSION_DIR", str(tmp_path))
    api._store.cache_clear()
    yield
    api._store.cache_clear()


def test_get_serves_the_phone_page():
    response = api.handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/"}, None)
    assert response["statusCode"] == 200
    assert response["headers"]["content-type"].startswith("text/html")
    assert "<html" in response["body"].lower()


def test_a_full_grid_passes_and_a_missing_dot_needs_repair():
    status, full = _post({"session": None, "image": _jpeg(_scene()), "dot_mm": DOT_MM})
    assert status == 200
    assert (full["action"], full["rule"], full["frame"], full["dots_found"]) == ("pass", "all_clear", 1, 16)
    assert uuid.UUID(full["session"])
    assert cv2.imdecode(np.frombuffer(base64.b64decode(full["image"]), np.uint8), cv2.IMREAD_COLOR) is not None

    status, gap = _post({"session": None, "image": _jpeg(_scene(omit=((1, 2),))), "dot_mm": DOT_MM})
    assert status == 200
    assert (gap["action"], gap["rule"]) == ("repair", "confirmed_gap")
    assert RULE_MM < gap["largest_gap_mm"] < 70


def test_a_grid_too_sparse_to_group_still_fails_the_rule():
    # 90 mm spacing leaves 71 mm between dot edges: every gap breaks the 2-inch rule, so the area
    # checked must span the whole grid, not shrink to the single largest dot group.
    status, body = _post({"session": None, "image": _jpeg(_scene(spacing_mm=90.0)), "dot_mm": DOT_MM})
    assert status == 200
    assert (body["action"], body["rule"]) == ("repair", "confirmed_gap")
    assert body["largest_gap_mm"] > RULE_MM


def test_changing_the_dot_size_inside_a_scan_is_refused():
    _status, first = _post({"session": None, "image": _jpeg(_scene()), "dot_mm": DOT_MM})
    status, body = _post({"session": first["session"], "image": _jpeg(_scene()), "dot_mm": 6.35})
    assert status == 400 and "new scan" in body["error"]


def test_a_full_scan_session_is_refused(monkeypatch):
    monkeypatch.setattr(api, "MAX_SESSION_PHOTOS", 1)
    _status, first = _post({"session": None, "image": _jpeg(_scene()), "dot_mm": DOT_MM})
    status, body = _post({"session": first["session"], "image": _jpeg(_scene()), "dot_mm": DOT_MM})
    assert status == 400 and "new scan" in body["error"]


def test_a_session_carries_its_frames_across_requests():
    _status, first = _post({"session": None, "image": _jpeg(_scene()), "dot_mm": DOT_MM})
    status, second = _post({"session": first["session"], "image": _jpeg(_scene()), "dot_mm": DOT_MM})
    assert status == 200
    assert (second["session"], second["frame"]) == (first["session"], 2)


def test_a_photo_without_the_sheet_asks_for_a_retake_without_a_session():
    blank = np.full((480, 640, 3), 150, np.uint8)
    status, body = _post({"session": None, "image": _jpeg(blank), "dot_mm": DOT_MM})
    assert status == 200
    assert (body["action"], body["rule"], body["session"], body["image"]) == (
        "recapture", "board_not_found", None, None,
    )


@pytest.mark.parametrize(
    "body",
    [
        {"session": None, "dot_mm": DOT_MM},
        {"session": None, "image": "bm90IGFuIGltYWdl", "dot_mm": DOT_MM},
        {"session": None, "image": "!!!!", "dot_mm": DOT_MM},
        {"session": None, "image": "====", "dot_mm": DOT_MM},
        {"session": "../../etc", "image": "x", "dot_mm": DOT_MM},
        {"session": None, "image": "x", "dot_mm": 500},
    ],
)
def test_bad_requests_get_a_specific_400(body):
    status, payload = _post(body)
    assert status == 400 and payload["error"]


def test_unknown_routes_are_404():
    response = api.handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/admin"}, None)
    assert response["statusCode"] == 404
