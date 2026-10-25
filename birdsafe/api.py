"""AWS Lambda entry behind a function URL: GET / serves the phone page, POST /scan folds one photo
into a scan session and answers with the decision and an annotated image.

Session state is the fused masks and the trace, never the photo. It lives in S3 when
SESSION_BUCKET is set, otherwise in SESSION_DIR (local runs and tests).
"""

import base64
import functools
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from birdsafe.calib import Rejected, rectify
from birdsafe.fusion import Observation
from birdsafe.markers import DOT_MM
from birdsafe.policy import trace_from_jsonl, trace_to_jsonl
from birdsafe.render import patch_canvas, render_step
from birdsafe.scan import ScanState, scan_step

# Lambda function URLs cap a request at 6 MB; the page sends a 2400 px JPEG of 1 to 2 MB.
MAX_IMAGE_B64 = 5_500_000
# Each request reloads and fuses the whole session again, so a public URL needs a ceiling. The
# policy stops after 3 stalled retakes; 8 leaves room for repairs and rescans.
MAX_SESSION_PHOTOS = 8
DOT_MM_RANGE = (3.0, 60.0)
PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"
NO_DOTS_MESSAGE = (
    "No markings were found beside the calibration sheet. Keep the whole sheet and the marked "
    "area in the photo, then retake it."
)

log = logging.getLogger()
log.setLevel(logging.INFO)


class BadRequest(ValueError):
    pass


@functools.cache
def _store():
    bucket = os.environ.get("SESSION_BUCKET")
    if bucket:
        import boto3  # ponytail: in the Lambda runtime only; local runs use a folder.

        s3 = boto3.client("s3")

        def get(key):
            try:
                return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            except s3.exceptions.NoSuchKey:
                return None

        def put(key, data):
            s3.put_object(Bucket=bucket, Key=key, Body=data)

        return get, put

    root = Path(os.environ.get("SESSION_DIR", "/tmp/birdsafe-sessions"))

    def get_local(key):
        path = root / key
        return path.read_bytes() if path.exists() else None

    def put_local(key, data):
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    return get_local, put_local


def _save(session: str, state: ScanState) -> None:
    _get, put = _store()
    arrays = {}
    for i, obs in enumerate(state.observations):
        arrays[f"markers_{i}"] = obs.markers
        arrays[f"unknown_{i}"] = obs.unknown
        arrays[f"origin_{i}"] = np.asarray(obs.board_origin_mm)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    meta = {
        "canvas_mm": state.canvas_mm,
        "mm_per_px": state.mm_per_px,
        "dot_mm": state.dot_mm,
        "frames": len(state.observations),
        "trace": trace_to_jsonl(state.trace),
    }
    put(f"sessions/{session}/obs.npz", buffer.getvalue())
    put(f"sessions/{session}/meta.json", json.dumps(meta).encode())


def _load(session: str) -> ScanState | None:
    get, _put = _store()
    raw = get(f"sessions/{session}/meta.json")
    if raw is None:
        return None
    meta = json.loads(raw)
    data = np.load(io.BytesIO(get(f"sessions/{session}/obs.npz")), allow_pickle=False)
    observations = tuple(
        Observation(
            data[f"markers_{i}"],
            data[f"unknown_{i}"],
            tuple(float(v) for v in data[f"origin_{i}"]),
            meta["mm_per_px"],
        )
        for i in range(meta["frames"])
    )
    return ScanState(
        canvas_mm=tuple(meta["canvas_mm"]),
        mm_per_px=meta["mm_per_px"],
        dot_mm=meta["dot_mm"],
        observations=observations,
        trace=tuple(trace_from_jsonl(meta["trace"])),
    )


def _parse(body: dict) -> tuple[np.ndarray, float, str | None]:
    encoded = body.get("image")
    if not isinstance(encoded, str) or not encoded:
        raise BadRequest("No photo was sent.")
    if len(encoded) > MAX_IMAGE_B64:
        raise BadRequest("The photo is too large. Send one under 4 MB.")
    try:
        raw = base64.b64decode(encoded, validate=True)
        # cv2.error also covers OPENCV_IO_MAX_IMAGE_PIXELS (set in deploy.sh), which stops a small
        # compressed file from decoding into gigabytes.
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    except (ValueError, cv2.error) as err:
        raise BadRequest("The photo could not be read. Send a JPEG or PNG.") from err
    if image is None:
        raise BadRequest("The photo could not be read. Send a JPEG or PNG.")

    try:
        dot_mm = float(body.get("dot_mm", DOT_MM))
    except (TypeError, ValueError) as err:
        raise BadRequest("The dot size must be a number of millimetres.") from err
    if not DOT_MM_RANGE[0] <= dot_mm <= DOT_MM_RANGE[1]:
        raise BadRequest(f"The dot size must be {DOT_MM_RANGE[0]:.0f} to {DOT_MM_RANGE[1]:.0f} mm.")

    session = body.get("session")
    if session is not None:
        try:
            # Also keeps the S3 key to one canonical shape.
            session = str(uuid.UUID(str(session)))
        except ValueError as err:
            raise BadRequest("The scan session id is not valid. Start a new scan.") from err
    return image, dot_mm, session


def _jpeg_b64(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])[1]).decode()


def scan(body: dict) -> dict:
    started = time.perf_counter()
    image, dot_mm, session = _parse(body)
    state = _load(session) if session else None
    if state is not None and abs(state.dot_mm - dot_mm) > 1e-6:
        raise BadRequest("The dot size changed during this scan. Start a new scan to use a different size.")
    if state is not None and len(state.trace) >= MAX_SESSION_PHOTOS:
        raise BadRequest(f"This scan already has {len(state.trace)} photos. Start a new scan.")
    capture = rectify(image)

    if state is None:
        canvas = None if isinstance(capture, Rejected) else patch_canvas(capture, dot_mm)
        if canvas is None:
            # Nothing to anchor a session to yet. A rejected capture still goes through the policy
            # (on a placeholder canvas) so the user gets its specific retake message.
            if isinstance(capture, Rejected):
                probe = scan_step(ScanState((0.0, 0.0, 1.0, 1.0), dot_mm=dot_mm), capture).trace[-1]
                rule, message = probe.decision.rule, probe.decision.message
            else:
                rule, message = "no_markings", NO_DOTS_MESSAGE
            return {
                "session": None, "frame": 0, "action": "recapture", "rule": rule,
                "message": message, "largest_gap_mm": 0.0, "unknown_fraction": 0.0,
                "dots_found": 0,
                "image": None if isinstance(capture, Rejected) else _jpeg_b64(capture.image),
            }
        session = str(uuid.uuid4())
        state = ScanState(canvas_mm=canvas, dot_mm=dot_mm)

    state = scan_step(state, capture)
    _save(session, state)
    rendered, dots = (None, 0) if isinstance(capture, Rejected) else render_step(capture, state)
    record = state.trace[-1]
    result = {
        "session": session,
        "frame": len(state.trace),
        "action": record.decision.action,
        "rule": record.decision.rule,
        "message": record.decision.message,
        "largest_gap_mm": round(record.metrics.largest_gap_mm, 1),
        "unknown_fraction": round(record.metrics.unknown_fraction, 4),
        "dots_found": dots,
    }
    # One structured line per scan: CloudWatch Logs Insights can chart decisions and latency.
    log.info(json.dumps({**result, "latency_ms": round((time.perf_counter() - started) * 1000)}))
    return {**result, "image": None if rendered is None else _jpeg_b64(rendered)}


def _respond(status: int, body: str, content_type: str) -> dict:
    return {"statusCode": status, "headers": {"content-type": content_type}, "body": body}


def _json(status: int, payload: dict) -> dict:
    return _respond(status, json.dumps(payload), "application/json")


def handler(event, _context):
    http = event.get("requestContext", {}).get("http", {})
    method, path = http.get("method", "GET"), event.get("rawPath", "/")
    if method == "GET" and path in ("/", "/index.html"):
        return _respond(200, PAGE.read_text(encoding="utf-8"), "text/html; charset=utf-8")
    if method != "POST" or path != "/scan":
        return _json(404, {"error": "Not found."})
    try:
        raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode()
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise BadRequest("Send a JSON object.")
        return _json(200, scan(body))
    except BadRequest as err:
        return _json(400, {"error": str(err)})
    except json.JSONDecodeError:
        return _json(400, {"error": "Send a JSON object."})
    except Exception:
        log.exception("scan failed")
        return _json(500, {"error": "The checker hit an internal error. Try another photo."})
