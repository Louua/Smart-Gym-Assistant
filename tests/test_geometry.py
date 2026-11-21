"""Tests for vector trigonometry and geometry functions."""

import math
import pytest
from core.geometry import angle_between, trunk_inclination, select_side
from core.models import Keypoint, PoseFrame


def test_angle_between_right_angle():
    # 90-degree right triangle: a=(0, 1), b=(0, 0), c=(1, 0)
    angle = angle_between((0.0, 1.0), (0.0, 0.0), (1.0, 0.0))
    assert math.isclose(angle, 90.0, abs_tol=1e-5)


def test_angle_between_straight_line():
    # 180-degree straight angle: a=(-1, 0), b=(0, 0), c=(1, 0)
    angle = angle_between((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    assert math.isclose(angle, 180.0, abs_tol=1e-5)


def test_angle_between_45_degrees():
    # 45-degree angle: a=(1, 1), b=(0, 0), c=(1, 0)
    angle = angle_between((1.0, 1.0), (0.0, 0.0), (1.0, 0.0))
    assert math.isclose(angle, 45.0, abs_tol=1e-5)


def test_angle_between_zero_vector():
    # Zero length vector should return NaN and not raise ZeroDivisionError
    angle = angle_between((0.0, 0.0), (0.0, 0.0), (1.0, 1.0))
    assert math.isnan(angle)


def test_trunk_inclination():
    # Perfectly upright: hip at (100, 200), shoulder at (100, 100)
    # in image coords y increases downwards, so shoulder is higher than hip
    angle_upright = trunk_inclination((100.0, 100.0), (100.0, 200.0))
    assert math.isclose(angle_upright, 0.0, abs_tol=1e-5)

    # Completely horizontal lean forward: shoulder at (200, 200), hip at (100, 200)
    angle_horiz = trunk_inclination((200.0, 200.0), (100.0, 200.0))
    assert math.isclose(angle_horiz, 90.0, abs_tol=1e-5)


def test_select_side_override():
    dummy_frame = PoseFrame(frame_idx=0, timestamp_s=0.0, keypoints={})
    assert select_side(dummy_frame, side_override="right") == "right"
    assert select_side(dummy_frame, side_override="left") == "left"


def test_select_side_auto():
    # Left side has higher confidence
    kps = {
        "left_shoulder": Keypoint(0.4, 0.2, 0.95),
        "left_hip": Keypoint(0.4, 0.5, 0.95),
        "left_knee": Keypoint(0.4, 0.7, 0.95),
        "right_shoulder": Keypoint(0.6, 0.2, 0.2),
        "right_hip": Keypoint(0.6, 0.5, 0.2),
        "right_knee": Keypoint(0.6, 0.7, 0.2),
    }
    frame = PoseFrame(frame_idx=0, timestamp_s=0.0, keypoints=kps)
    assert select_side(frame, side_override="auto") == "left"
