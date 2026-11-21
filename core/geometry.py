"""Biomechanical geometry and vector calculations.

Provides pixel-aspect-correct 2D joint angle computations and side visibility resolution.
"""

from __future__ import annotations
import math
from typing import Sequence, Tuple
from core.models import Keypoint, PoseFrame


def angle_between(
    a: Tuple[float, float] | Sequence[float],
    b: Tuple[float, float] | Sequence[float],
    c: Tuple[float, float] | Sequence[float],
) -> float:
    """Calculate the interior planar angle at vertex `b` formed by points a-b-c in degrees.

    Formula:
        theta = arccos( (u . v) / (||u|| * ||v||) )
        where u = a - b, v = c - b

    Args:
        a: Coordinates of first point (x, y).
        b: Coordinates of vertex point (x, y).
        c: Coordinates of third point (x, y).

    Returns:
        Angle in degrees [0, 180], or math.nan if zero-length vector.
    """
    ux = a[0] - b[0]
    uy = a[1] - b[1]
    vx = c[0] - b[0]
    vy = c[1] - b[1]

    dot = ux * vx + uy * vy
    norm_u = math.sqrt(ux * ux + uy * uy)
    norm_v = math.sqrt(vx * vx + vy * vy)

    if norm_u < 1e-7 or norm_v < 1e-7:
        return math.nan

    cos_theta = dot / (norm_u * norm_v)
    # Numerical precision guard: clamp to [-1.0, 1.0]
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return math.degrees(math.acos(cos_theta))


def denormalize_point(kp: Keypoint, width: int, height: int) -> Tuple[float, float]:
    """Convert normalized [0, 1] keypoint coordinates into pixel coordinates.

    Documented rationale:
        In video analysis, camera sensors often have non-square aspect ratios (e.g. 16:9, 4:3).
        Computing angles on raw normalized coordinates skews vectors along the wider dimension.
        Multiplying by (width, height) restores true Euclidean planar geometry.
    """
    return (kp.x * width, kp.y * height)


def trunk_inclination(
    shoulder: Tuple[float, float] | Sequence[float],
    hip: Tuple[float, float] | Sequence[float],
) -> float:
    """Calculate the inclination of the torso vector relative to the vertical axis in degrees.

    In image coordinates, y increases downward.
    Vector from hip to shoulder:
        v = (shoulder_x - hip_x, shoulder_y - hip_y)
    Vertical upward vector:
        up = (0.0, -1.0)
    A perfectly vertical upright posture yields 0.0 degrees.
    Horizontal forward lean yields 90.0 degrees.
    """
    # Vector from hip to shoulder
    vx = shoulder[0] - hip[0]
    vy = shoulder[1] - hip[1]

    norm_v = math.sqrt(vx * vx + vy * vy)
    if norm_v < 1e-7:
        return math.nan

    # Vertical upward vector is (0, -1)
    # dot(v, (0, -1)) = -vy
    cos_theta = -vy / norm_v
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return math.degrees(math.acos(cos_theta))


def select_side(
    pose_frame: PoseFrame,
    side_override: str = "auto",
    recent_frames: Sequence[PoseFrame] | None = None,
) -> str:
    """Determine whether to track 'left' or 'right' body side.

    In profile/sagittal views, one side faces the camera while the other is partially occluded.
    Tracking the side with higher visibility confidence prevents noisy angle jumps.

    Args:
        pose_frame: Current pose frame.
        side_override: 'auto', 'left', or 'right'.
        recent_frames: Optional list of recent historical frames for smoothed confidence.

    Returns:
        'left' or 'right'.
    """
    if side_override in ("left", "right"):
        return side_override

    frames_to_eval = list(recent_frames) if recent_frames else []
    if pose_frame not in frames_to_eval:
        frames_to_eval.append(pose_frame)

    left_scores: list[float] = []
    right_scores: list[float] = []

    joints = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]

    for f in frames_to_eval:
        if not f.keypoints:
            continue
        l_vis = [f.keypoints[f"left_{j}"].visibility for j in joints if f"left_{j}" in f.keypoints]
        r_vis = [f.keypoints[f"right_{j}"].visibility for j in joints if f"right_{j}" in f.keypoints]

        if l_vis:
            left_scores.append(sum(l_vis) / len(l_vis))
        if r_vis:
            right_scores.append(sum(r_vis) / len(r_vis))

    avg_left = sum(left_scores) / len(left_scores) if left_scores else 0.0
    avg_right = sum(right_scores) / len(right_scores) if right_scores else 0.0

    return "left" if avg_left >= avg_right else "right"
