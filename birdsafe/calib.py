"""Metric rectification from a printed ChArUco board.

The board is the only source of physical scale. Without it there is no honest millimetre
measurement, so a failed detection returns a Rejected with its reason rather than a guess.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from tools.charuco_sheet import SQUARES_X, birdsafe_board

MIN_CORNERS = 6
CORNERS_PER_ROW = SQUARES_X - 1
# ponytail: 8000 px is 4 m at 0.5 mm/px; a bigger footprint means a view too oblique to
# measure.
MAX_SIDE_PX = 8000


@dataclass(frozen=True)
class Rectified:
    image: np.ndarray
    mm_per_px: float
    # Where the board's top-left corner sits in `image`. Every capture of one pane shares the
    # board plane, so subtracting this puts all captures in one frame without registration.
    board_origin_mm: tuple[float, float]
    reproj_rms_mm: float
    corners_found: int
    # True where the pixel came from the capture. warpPerspective fills the rest with black the
    # camera never saw.
    valid: np.ndarray
    # Capture resolution on the board plane along its coarsest axis.
    native_mm_per_px: float
    view_angle_deg: float


@dataclass(frozen=True)
class Rejected:
    reason: str


def _map_homogeneous(h_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Applies h_matrix to Nx2 points without dividing by w, so callers can inspect w's sign."""
    pts_h = np.hstack([points.astype(np.float64), np.ones((len(points), 1))])
    return (h_matrix @ pts_h.T).T


def _view_geometry(h_matrix: np.ndarray, dst: np.ndarray, mm_per_px: float) -> tuple[float, float]:
    """Returns (native_mm_per_px, view_angle_deg) at the centre of the corners the board gave.

    Without camera intrinsics this is a foreshortening estimate, not a measured pose.
    """
    centre = dst.mean(axis=0)
    step_px = 1.0 / mm_per_px
    plane_points = np.float32([centre, centre + [step_px, 0.0], centre + [0.0, step_px]])
    source = cv2.perspectiveTransform(
        plane_points.reshape(-1, 1, 2), np.linalg.inv(h_matrix)
    ).reshape(-1, 2)
    jacobian = (source[1:] - source[0]).T
    s1, s2 = np.linalg.svd(jacobian, compute_uv=False)
    return float(1.0 / s2), float(np.degrees(np.arccos(min(1.0, s2 / s1))))


def rectify(image: np.ndarray, mm_per_px: float = 0.5) -> Rectified | Rejected:
    board = birdsafe_board()
    corners, ids, _marker_corners, _marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if corners is None or ids is None or len(corners) < MIN_CORNERS:
        return Rejected("board_not_found")
    ids = ids.flatten()
    # Corners from a single board row or column are collinear, and a homography fitted to them
    # stretches the unseen axis arbitrarily.
    if len(set(ids // CORNERS_PER_ROW)) < 2 or len(set(ids % CORNERS_PER_ROW)) < 2:
        return Rejected("board_partly_hidden")

    # Board corner (i, j) sits at a known millimetre offset, which fixes the output scale.
    obj = board.getChessboardCorners()[ids][:, :2] * 1000.0 / mm_per_px
    src = corners.reshape(-1, 2).astype(np.float32)
    dst = obj.astype(np.float32)

    h_matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if h_matrix is None:
        return Rejected("board_not_found")

    fit = inliers.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src[fit].reshape(-1, 1, 2), h_matrix).reshape(-1, 2)
    rms_px = float(np.sqrt(np.mean(np.sum((projected - dst[fit]) ** 2, axis=1))))

    img_h, img_w = image.shape[:2]
    frame_corners = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

    # The homography's overall sign is arbitrary. Anchor it to a corner actually seen on the
    # board, since that point is known to sit on the visible, in-front-of-camera side of the plane.
    ref_w = float(_map_homogeneous(h_matrix, src[:1])[0, 2])
    if ref_w == 0:
        return Rejected("view_too_oblique")
    ref_sign = 1.0 if ref_w > 0 else -1.0

    mapped_h = _map_homogeneous(h_matrix, frame_corners)
    w = mapped_h[:, 2]
    if np.any(w == 0) or np.any(np.sign(w) != ref_sign):
        # A capture corner reaches the plane's horizon: it maps to infinity or behind the plane,
        # so no finite rectified view exists.
        return Rejected("view_too_oblique")
    mapped = mapped_h[:, :2] / w[:, None]

    min_xy = mapped.min(axis=0)
    max_xy = mapped.max(axis=0)
    out_w = round(float(max_xy[0] - min_xy[0]))
    out_h = round(float(max_xy[1] - min_xy[1]))
    if out_w > MAX_SIDE_PX or out_h > MAX_SIDE_PX:
        return Rejected("view_too_large")

    translate = np.float64([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]])
    full_matrix = translate @ h_matrix
    warped = cv2.warpPerspective(image, full_matrix, (out_w, out_h))
    valid = cv2.warpPerspective(
        np.full((img_h, img_w), 255, np.uint8), full_matrix, (out_w, out_h), flags=cv2.INTER_NEAREST
    ) > 0
    native_mm_per_px, view_angle_deg = _view_geometry(h_matrix, dst, mm_per_px)

    return Rectified(
        image=warped,
        mm_per_px=mm_per_px,
        board_origin_mm=(float(-min_xy[0]) * mm_per_px, float(-min_xy[1]) * mm_per_px),
        reproj_rms_mm=rms_px * mm_per_px,
        corners_found=int(len(corners)),
        valid=valid,
        native_mm_per_px=native_mm_per_px,
        view_angle_deg=view_angle_deg,
    )
