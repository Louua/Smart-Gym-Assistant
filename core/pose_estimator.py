"""Module 2: Pose Estimation Layer.

Wraps MediaPipe Pose (legacy mp.solutions.pose API) to extract the 12 core biomechanical
landmarks and provides skeleton drawing with angle callouts and FSM status.
"""

from __future__ import annotations
from typing import Any, Tuple
import cv2
import numpy as np
import mediapipe as mp

from core.models import Keypoint, PoseFrame


# 12 Landmark names mapped to MediaPipe PoseLandmark indices
LANDMARK_MAP = {
    "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
}

# Skeleton connection pairs for drawing
SKELETON_CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Left arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Right arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


class PoseEstimator:
    """Wrapper around MediaPipe Pose estimation."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            static_image_mode=self.static_image_mode,
        )

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
    ) -> PoseFrame:
        """Run pose inference on a single BGR frame.

        Returns:
            PoseFrame with extracted 12 landmarks, or keypoints=None if no person detected.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._pose.process(rgb_frame)

        if not results.pose_landmarks:
            return PoseFrame(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                keypoints=None,
            )

        landmarks = results.pose_landmarks.landmark
        keypoints: dict[str, Keypoint] = {}

        for name, landmark_enum in LANDMARK_MAP.items():
            lm = landmarks[landmark_enum.value]
            keypoints[name] = Keypoint(
                x=float(lm.x),
                y=float(lm.y),
                visibility=float(lm.visibility),
            )

        return PoseFrame(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            keypoints=keypoints,
        )

    def draw_skeleton(
        self,
        bgr_image: np.ndarray,
        pose_frame: PoseFrame,
        active_side: str | None = None,
        angles: dict[str, float] | None = None,
        state: str | None = None,
        rep_count: int | None = None,
        feedback: list[str] | None = None,
    ) -> np.ndarray:
        """Overlay skeleton, keypoints, angles, and feedback on the image."""
        annotated = bgr_image.copy()
        h, w = annotated.shape[:2]

        if pose_frame.keypoints is None:
            # Display warning that no body is detected
            cv2.putText(
                annotated,
                "No person detected in frame",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return annotated

        kps = pose_frame.keypoints

        # Draw connections
        for p1_name, p2_name in SKELETON_CONNECTIONS:
            if p1_name in kps and p2_name in kps:
                kp1, kp2 = kps[p1_name], kps[p2_name]
                if kp1.visibility >= 0.3 and kp2.visibility >= 0.3:
                    pt1 = (int(kp1.x * w), int(kp1.y * h))
                    pt2 = (int(kp2.x * w), int(kp2.y * h))

                    # Color code active side (green) vs inactive side (cyan/gray)
                    color = (255, 200, 0)  # Default gold
                    if active_side:
                        if p1_name.startswith(active_side) and p2_name.startswith(active_side):
                            color = (0, 255, 0)  # Bright green for active limb
                        elif not p1_name.startswith(active_side) and not p2_name.startswith(active_side):
                            color = (180, 180, 180)  # Muted for far limb

                    cv2.line(annotated, pt1, pt2, color, 3, cv2.LINE_AA)

        # Draw joint nodes
        for name, kp in kps.items():
            if kp.visibility >= 0.3:
                pt = (int(kp.x * w), int(kp.y * h))
                node_color = (0, 255, 0) if active_side and name.startswith(active_side) else (0, 140, 255)
                cv2.circle(annotated, pt, 5, node_color, -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Heads-Up-Display (HUD) overlay banner
        # Top HUD bar: Semi-transparent background
        hud_h = 70
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # State and Rep Count
        if state is not None:
            cv2.putText(
                annotated,
                f"STATE: {state.upper()}",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if rep_count is not None:
            rep_text = f"REPS: {rep_count}"
            cv2.putText(
                annotated,
                rep_text,
                (w - 180, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Angles readout (middle/bottom HUD or near joints)
        if angles:
            angle_str = " | ".join(f"{k.capitalize()}: {v:.1f}°" for k, v in angles.items() if not np.isnan(v))
            cv2.putText(
                annotated,
                angle_str,
                (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Feedback warnings (red banner)
        if feedback:
            fb_y = hud_h + 35
            for msg in feedback:
                # Text with dark shadow for legibility
                cv2.putText(
                    annotated,
                    f"⚠️ {msg}",
                    (22, fb_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"⚠️ {msg}",
                    (20, fb_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 80, 255),
                    2,
                    cv2.LINE_AA,
                )
                fb_y += 30

        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
