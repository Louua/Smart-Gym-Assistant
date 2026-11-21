"""Module 3: Temporal Tracking Layer.

Implements the 6-step temporal pipeline:
1. Detect points per image (from Module 2).
2. Extract coordinates.
3. Link associations across consecutive frames.
4. Maintain trajectory history.
5. Mathematical smoothing (coordinate EMA and optional angle filters).
6. Expose trajectory history for exercise cycle analysis.

Also handles missing-joint carry-forward up to max_hold_frames.
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Any
import numpy as np
from scipy.signal import savgol_filter, medfilt

from core.models import Keypoint, PoseFrame


class TemporalTracker:
    """Maintains temporal landmark trajectories and applies noise filters."""

    def __init__(
        self,
        alpha: float = 0.4,
        smoothing_method: str = "ema",
        window_size: int = 7,
        min_visibility: float = 0.5,
        max_hold_frames: int = 10,
        buffer_len: int = 120,
    ) -> None:
        self.alpha = float(alpha)
        self.smoothing_method = smoothing_method
        self.window_size = int(window_size)
        self.min_visibility = float(min_visibility)
        self.max_hold_frames = int(max_hold_frames)
        self.buffer_len = int(buffer_len)

        # Smoothed coordinates: {joint_name: (x, y)}
        self._smoothed_coords: Dict[str, tuple[float, float]] = {}
        # Consecutive dropped/low-confidence frame count: {joint_name: int}
        self._missing_counts: Dict[str, int] = {}
        # Last observed visibility: {joint_name: float}
        self._visibilities: Dict[str, float] = {}

        # Trajectory history buffers: {joint_name: deque([(t, x, y, vis)])}
        self._trajectories: Dict[str, deque[tuple[float, float, float, float]]] = {}
        # Angle history buffer: {angle_name: deque([angle])}
        self._angle_history: Dict[str, deque[float]] = {}
        # Full session storage
        self.full_history: List[PoseFrame] = []
        self.dropped_frames: int = 0

    def update(self, raw_frame: PoseFrame) -> PoseFrame:
        """Update trajectories with a new raw pose frame and return smoothed PoseFrame."""
        t = raw_frame.timestamp_s

        if raw_frame.keypoints is None:
            self.dropped_frames += 1
            # Age all held coordinates
            smoothed_kps: Dict[str, Keypoint] = {}
            for joint, count in list(self._missing_counts.items()):
                self._missing_counts[joint] = count + 1
                if self._missing_counts[joint] <= self.max_hold_frames and joint in self._smoothed_coords:
                    x, y = self._smoothed_coords[joint]
                    vis = max(0.0, self._visibilities.get(joint, 0.0) * 0.9)  # Decay confidence
                    smoothed_kps[joint] = Keypoint(x=x, y=y, visibility=vis)
                else:
                    self._smoothed_coords.pop(joint, None)

            smoothed_frame = PoseFrame(
                frame_idx=raw_frame.frame_idx,
                timestamp_s=t,
                keypoints=smoothed_kps if smoothed_kps else None,
            )
            self.full_history.append(smoothed_frame)
            return smoothed_frame

        smoothed_kps: Dict[str, Keypoint] = {}

        for joint, kp in raw_frame.keypoints.items():
            if joint not in self._trajectories:
                self._trajectories[joint] = deque(maxlen=self.buffer_len)

            if kp.visibility >= self.min_visibility:
                # Valid high-confidence landmark
                self._missing_counts[joint] = 0
                self._visibilities[joint] = kp.visibility

                if joint not in self._smoothed_coords:
                    # Initial position
                    smooth_x, smooth_y = kp.x, kp.y
                else:
                    # Exponential Moving Average (EMA)
                    prev_x, prev_y = self._smoothed_coords[joint]
                    smooth_x = self.alpha * kp.x + (1.0 - self.alpha) * prev_x
                    smooth_y = self.alpha * kp.y + (1.0 - self.alpha) * prev_y

                self._smoothed_coords[joint] = (smooth_x, smooth_y)
                smoothed_kps[joint] = Keypoint(x=smooth_x, y=smooth_y, visibility=kp.visibility)
                self._trajectories[joint].append((t, smooth_x, smooth_y, kp.visibility))
            else:
                # Low confidence or occluded joint
                count = self._missing_counts.get(joint, 0) + 1
                self._missing_counts[joint] = count

                if count <= self.max_hold_frames and joint in self._smoothed_coords:
                    # Hold previous position
                    smooth_x, smooth_y = self._smoothed_coords[joint]
                    decayed_vis = max(0.0, self._visibilities.get(joint, 0.0) * 0.9)
                    smoothed_kps[joint] = Keypoint(x=smooth_x, y=smooth_y, visibility=decayed_vis)
                    self._trajectories[joint].append((t, smooth_x, smooth_y, decayed_vis))
                else:
                    # Expired hold: drop joint
                    self._smoothed_coords.pop(joint, None)

        smoothed_frame = PoseFrame(
            frame_idx=raw_frame.frame_idx,
            timestamp_s=t,
            keypoints=smoothed_kps if smoothed_kps else None,
        )
        self.full_history.append(smoothed_frame)
        return smoothed_frame

    def smooth_angle(self, angle_name: str, current_angle: float) -> float:
        """Apply configured angle-level temporal filtering (moving median or savgol)."""
        if np.isnan(current_angle):
            return current_angle

        if angle_name not in self._angle_history:
            self._angle_history[angle_name] = deque(maxlen=max(self.window_size * 2, 15))

        buf = self._angle_history[angle_name]
        buf.append(current_angle)

        if self.smoothing_method == "median":
            w = self.window_size if self.window_size % 2 == 1 else self.window_size + 1
            if len(buf) >= w:
                window_vals = list(buf)[-w:]
                return float(np.median(window_vals))
            return current_angle

        elif self.smoothing_method == "savgol":
            w = self.window_size if self.window_size % 2 == 1 else self.window_size + 1
            if len(buf) >= w:
                window_vals = list(buf)[-w:]
                try:
                    smoothed = savgol_filter(window_vals, window_length=w, polyorder=2)
                    return float(smoothed[-1])
                except Exception:
                    return current_angle
            return current_angle

        # Default: coordinate EMA already handled noise; pass through or simple avg
        return current_angle

    def trajectory(self, joint: str) -> List[Dict[str, float]]:
        """Return trajectory records for a joint: [{'t': t, 'x': x, 'y': y, 'vis': vis}]."""
        if joint not in self._trajectories:
            return []
        return [
            {"t": pt[0], "x": pt[1], "y": pt[2], "vis": pt[3]}
            for pt in self._trajectories[joint]
        ]

    def reset(self) -> None:
        """Reset internal buffers for a new session."""
        self._smoothed_coords.clear()
        self._missing_counts.clear()
        self._visibilities.clear()
        self._trajectories.clear()
        self._angle_history.clear()
        self.full_history.clear()
        self.dropped_frames = 0
