"""Tests for Bicep Curl FSM, rep counting, and shoulder compensation detection."""

import pytest
import numpy as np
from core.exercises.bicep_curl import BicepCurlExercise
from core.models import Keypoint, PoseFrame


def make_curl_frame(
    idx: int,
    t: float,
    elbow_angle_deg: float,
    shoulder_y: float = 0.2,
) -> PoseFrame:
    """Helper creating synthetic PoseFrame with specified elbow angle and shoulder position."""
    rad = np.radians(elbow_angle_deg)
    # Shoulder at (0.3, shoulder_y), Elbow at (0.3, shoulder_y + 0.2)
    # Wrist rotates around elbow
    wrist_x = 0.3 + 0.2 * np.sin(np.pi - rad)
    wrist_y = (shoulder_y + 0.2) + 0.2 * np.cos(np.pi - rad)

    kps = {
        "right_shoulder": Keypoint(0.3, shoulder_y, 0.9),
        "right_elbow": Keypoint(0.3, shoulder_y + 0.2, 0.9),
        "right_wrist": Keypoint(float(wrist_x), float(wrist_y), 0.9),
        "right_hip": Keypoint(0.3, shoulder_y + 0.5, 0.9),
    }
    return PoseFrame(frame_idx=idx, timestamp_s=t, keypoints=kps)


def test_curl_clean_reps():
    curl = BicepCurlExercise()
    t = 0.0
    frame_idx = 0

    # 3 full repetitions: 160 -> 40 -> 160
    for _ in range(3):
        for angle in np.linspace(160, 40, 15):
            f = make_curl_frame(frame_idx, t, float(angle))
            curl.update(f, 1000, 1000, "right")
            t += 0.05
            frame_idx += 1

        for angle in np.linspace(40, 160, 15):
            f = make_curl_frame(frame_idx, t, float(angle))
            curl.update(f, 1000, 1000, "right")
            t += 0.05
            frame_idx += 1

    assert curl.rep_count == 3
    assert curl.rejected_attempts == 0


def test_curl_incomplete_flexion_rejected():
    curl = BicepCurlExercise()
    t = 0.0
    frame_idx = 0

    # Partial curl: starts at 160, reaches only 80 (fails to reach <= 50), returns to 155
    for angle in np.linspace(160, 80, 12):
        f = make_curl_frame(frame_idx, t, float(angle))
        curl.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1

    for angle in np.linspace(80, 155, 12):
        f = make_curl_frame(frame_idx, t, float(angle))
        curl.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1

    assert curl.rep_count == 0
    assert curl.rejected_attempts == 1
    assert any("incomplete_flexion" in r.issues for r in curl.reps if not r.valid)


def test_curl_shoulder_drift_compensation():
    curl = BicepCurlExercise()
    t = 0.0
    frame_idx = 0

    # Rep with deliberate excessive shoulder swing: starts at y=0.2, drifts to y=0.35
    for angle in np.linspace(160, 40, 15):
        drift = 0.2 + (0.15 * (160 - angle) / 120.0)
        f = make_curl_frame(frame_idx, t, float(angle), shoulder_y=drift)
        curl.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1

    for angle in np.linspace(40, 160, 15):
        f = make_curl_frame(frame_idx, t, float(angle), shoulder_y=0.2)
        curl.update(f, 1000, 1000, "right")
        t += 0.05
        frame_idx += 1

    assert curl.rep_count == 1
    # Should flag shoulder compensation in issues
    assert "shoulder_compensation" in curl.reps[0].issues
