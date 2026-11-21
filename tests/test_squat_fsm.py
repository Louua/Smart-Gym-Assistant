"""Tests for Squat FSM state progression, repetition counting, and rejection."""

import pytest
import numpy as np
from core.exercises.squat import SquatExercise
from core.models import Keypoint, PoseFrame


def make_squat_frame(idx: int, t: float, knee_angle_deg: float) -> PoseFrame:
    """Helper creating synthetic PoseFrame with exact knee angle."""
    # Hip at (100, 100), Knee at (100, 200).
    # Angle at Knee: vector u = Hip - Knee = (0, -100).
    # Angle theta: vector v = Ankle - Knee.
    # We want angle between u and v to be knee_angle_deg.
    # u direction is (0, -1). Rotated by theta:
    rad = np.radians(knee_angle_deg)
    # When rad=180 (straight leg), Ankle is at (100, 300) -> v = (0, 100).
    # Angle between (0, -1) and (0, 1) is 180.
    # When bent forward, ankle moves relative to knee:
    ankle_x = 100.0 + 100.0 * np.sin(np.pi - rad)
    ankle_y = 200.0 + 100.0 * np.cos(np.pi - rad)

    kps = {
        "right_shoulder": Keypoint(0.1, 0.05, 0.9),
        "right_hip": Keypoint(0.1, 0.1, 0.9),
        "right_knee": Keypoint(0.1, 0.2, 0.9),
        "right_ankle": Keypoint(ankle_x / 1000.0, ankle_y / 1000.0, 0.9),
    }
    return PoseFrame(frame_idx=idx, timestamp_s=t, keypoints=kps)


def test_squat_5_clean_reps():
    squat = SquatExercise()
    t = 0.0
    frame_idx = 0

    # 5 full repetitions: 170 -> 90 -> 170
    for _ in range(5):
        # Standing -> Descending -> Bottom
        for angle in np.linspace(170, 90, 15):
            f = make_squat_frame(frame_idx, t, float(angle))
            squat.update(f, image_width=1000, image_height=1000, side_override="right")
            t += 0.05
            frame_idx += 1

        # Bottom -> Ascending -> Standing
        for angle in np.linspace(90, 170, 15):
            f = make_squat_frame(frame_idx, t, float(angle))
            squat.update(f, image_width=1000, image_height=1000, side_override="right")
            t += 0.05
            frame_idx += 1

    assert squat.rep_count == 5
    assert squat.rejected_attempts == 0
    assert len(squat.reps) == 5
    assert all(r.valid for r in squat.reps)


def test_squat_shallow_reversal_rejected():
    squat = SquatExercise()
    t = 0.0
    frame_idx = 0

    # 1 valid rep
    for angle in np.linspace(170, 90, 12):
        f = make_squat_frame(frame_idx, t, float(angle))
        squat.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1
    for angle in np.linspace(90, 170, 12):
        f = make_squat_frame(frame_idx, t, float(angle))
        squat.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1

    # 1 shallow attempt: reverses at 130 deg (never reaches bottom <= 100)
    for angle in np.linspace(170, 130, 10):
        f = make_squat_frame(frame_idx, t, float(angle))
        squat.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1
    for angle in np.linspace(130, 165, 10):
        f = make_squat_frame(frame_idx, t, float(angle))
        squat.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1

    assert squat.rep_count == 1
    assert squat.rejected_attempts == 1
    assert any("insufficient_depth" in r.issues for r in squat.reps if not r.valid)


def test_squat_hysteresis_prevents_flicker():
    squat = SquatExercise()
    t = 0.0
    frame_idx = 0

    # Drop to 95 (BOTTOM)
    for angle in [170, 155, 145, 130, 115, 95]:
        f = make_squat_frame(frame_idx, t, float(angle))
        squat.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1

    assert squat.current_state == "BOTTOM"

    # Small fluctuation around 102-105 degrees (below 100 + hysteresis 10 = 110)
    for angle in [102, 105, 103, 104]:
        f = make_squat_frame(frame_idx, t, float(angle))
        res = squat.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1
        assert res.state == "BOTTOM", "Hysteresis must hold BOTTOM state against small jitter"
