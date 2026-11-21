"""Tests for TemporalTracker smoothing and occlusion hold."""

import numpy as np
from core.tracker import TemporalTracker
from core.models import Keypoint, PoseFrame


def test_ema_reduces_noise_variance():
    # EMA provably reduces the variance of an observation with zero-mean noise
    tracker = TemporalTracker(alpha=0.3)
    np.random.seed(42)
    n_points = 100
    true_val = 1.0
    noise = np.random.normal(0, 0.2, n_points)
    noisy_signal = true_val + noise

    smoothed_vals = []
    for i in range(n_points):
        kps = {"left_knee": Keypoint(x=float(noisy_signal[i]), y=0.5, visibility=0.9)}
        frame = PoseFrame(frame_idx=i, timestamp_s=float(i * 0.033), keypoints=kps)
        smooth_frame = tracker.update(frame)
        smoothed_vals.append(smooth_frame.keypoints["left_knee"].x)

    # Discard initial warm-up (first 10 samples)
    noisy_var = np.var(noisy_signal[10:])
    smooth_var = np.var(smoothed_vals[10:])

    # Smoothed variance must be substantially lower than raw noise variance
    assert smooth_var < (noisy_var * 0.6)


def test_missing_joint_hold_logic():
    tracker = TemporalTracker(min_visibility=0.5, max_hold_frames=3)

    # Frame 0: Valid point
    f0 = PoseFrame(0, 0.0, {"left_knee": Keypoint(0.5, 0.5, 0.9)})
    sf0 = tracker.update(f0)
    assert "left_knee" in sf0.keypoints

    # Frames 1, 2, 3: Occluded/missing (visibility=0.1) -> Should be held
    for i in range(1, 4):
        f = PoseFrame(i, float(i), {"left_knee": Keypoint(0.0, 0.0, 0.1)})
        sf = tracker.update(f)
        assert "left_knee" in sf.keypoints, f"Frame {i} should be held"
        assert sf.keypoints["left_knee"].x == 0.5

    # Frame 4: Max hold frames exceeded (3) -> Joint must be dropped
    f4 = PoseFrame(4, 4.0, {"left_knee": Keypoint(0.0, 0.0, 0.1)})
    sf4 = tracker.update(f4)
    assert sf4.keypoints is None or "left_knee" not in sf4.keypoints
