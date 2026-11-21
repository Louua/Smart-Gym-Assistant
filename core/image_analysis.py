"""Module: Single-Image Exercise & Posture Analysis.

Analyzes a static uploaded image using MediaPipe Pose (static_image_mode=True)
and Euclidean joint geometry to identify exercise posture, compute angles,
and generate actionable explanations and coaching solutions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import math
import numpy as np

from core.pose_estimator import PoseEstimator
from core.geometry import angle_between, denormalize_point, trunk_inclination, select_side


@dataclass
class ImageAnalysisResult:
    """Structured result for single-image exercise analysis."""

    annotated_bgr: np.ndarray
    detected: bool
    side_used: str
    detected_exercise: str
    posture_summary: str
    angles: Dict[str, float] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)
    solutions: List[str] = field(default_factory=list)


def analyze_exercise_image(
    bgr_image: np.ndarray,
    exercise_hint: str = "Auto-Detect",
    side_override: str = "auto",
) -> ImageAnalysisResult:
    """Run pose estimation and biomechanical analysis on a single BGR image."""
    h, w = bgr_image.shape[:2]
    estimator = PoseEstimator(static_image_mode=True)
    try:
        pose_frame = estimator.process_frame(bgr_image, frame_idx=0, timestamp_s=0.0)
        if pose_frame.keypoints is None:
            return ImageAnalysisResult(
                annotated_bgr=bgr_image.copy(),
                detected=False,
                side_used="none",
                detected_exercise="Unknown",
                posture_summary="No person was detected clearly in the uploaded image.",
                explanations=[
                    "MediaPipe Pose could not locate full-body landmarks in this frame."
                ],
                solutions=[
                    "Ensure your full body (head to ankles) is visible inside the frame.",
                    "Use bright, even lighting and avoid strong backlighting behind you.",
                    "Try a clear side or 45-degree angle against a plain background.",
                ],
            )

        side = select_side(pose_frame, side_override=side_override)
        kps = pose_frame.keypoints

        def pt(joint: str) -> tuple[float, float]:
            return denormalize_point(kps[f"{side}_{joint}"], w, h)

        shoulder = pt("shoulder")
        elbow = pt("elbow")
        wrist = pt("wrist")
        hip = pt("hip")
        knee = pt("knee")
        ankle = pt("ankle")

        knee_angle = angle_between(hip, knee, ankle)
        hip_angle = angle_between(shoulder, hip, knee)
        elbow_angle = angle_between(shoulder, elbow, wrist)
        trunk_lean = trunk_inclination(shoulder, hip)

        angles = {
            "knee_angle": round(knee_angle, 1) if not math.isnan(knee_angle) else 0.0,
            "hip_angle": round(hip_angle, 1) if not math.isnan(hip_angle) else 0.0,
            "elbow_angle": round(elbow_angle, 1) if not math.isnan(elbow_angle) else 0.0,
            "trunk_lean": round(trunk_lean, 1) if not math.isnan(trunk_lean) else 0.0,
        }

        # Determine exercise context
        if exercise_hint in ("Squats", "Bicep Curls"):
            detected_exercise = exercise_hint
        else:
            if angles["knee_angle"] < 145.0:
                detected_exercise = "Squats"
            elif angles["elbow_angle"] < 135.0:
                detected_exercise = "Bicep Curls"
            else:
                detected_exercise = "Standing / Neutral Posture"

        explanations: List[str] = []
        solutions: List[str] = []

        if detected_exercise == "Squats":
            if angles["knee_angle"] <= 100.0:
                posture_summary = "Bottom / Deep Squat Position Detected"
                explanations.append(
                    f"Knee angle is {angles['knee_angle']}° (<= 100°), indicating full squat depth."
                )
            elif angles["knee_angle"] < 150.0:
                posture_summary = "Mid-Squat / Descending Position Detected"
                explanations.append(
                    f"Knee angle is {angles['knee_angle']}° — partial squat depth (target <= 100° for full depth)."
                )
                solutions.append(
                    "Sit your hips back and down slightly further if your knees and ankles feel comfortable."
                )
            else:
                posture_summary = "Standing / Top Squat Position Detected"
                explanations.append(
                    f"Knee angle is {angles['knee_angle']}° (>= 160°), indicating full upright extension."
                )

            if angles["trunk_lean"] > 45.0:
                explanations.append(
                    f"Forward trunk lean is {angles['trunk_lean']}° (> 45° threshold), placing extra load on the lower back."
                )
                solutions.append(
                    "Keep your chest proud, brace your core, and try elevating your heels slightly or widening your stance."
                )
            else:
                explanations.append(
                    f"Torso alignment looks balanced with {angles['trunk_lean']}° trunk lean (<= 45°)."
                )

        elif detected_exercise == "Bicep Curls":
            if angles["elbow_angle"] <= 50.0:
                posture_summary = "Peak Flexion (Top of Bicep Curl) Detected"
                explanations.append(
                    f"Elbow angle is {angles['elbow_angle']}° (<= 50°), showing full bicep contraction."
                )
            elif angles["elbow_angle"] < 140.0:
                posture_summary = "Mid-Curl Position Detected"
                explanations.append(
                    f"Elbow angle is {angles['elbow_angle']}° — mid-range of the curl (peak target <= 50°, full extension >= 150°)."
                )
                solutions.append(
                    "Keep your upper arm pinned to your side and curl smoothly up toward your shoulder without swinging."
                )
            else:
                posture_summary = "Extended Arm (Start/Finish of Curl) Detected"
                explanations.append(
                    f"Elbow angle is {angles['elbow_angle']}° (>= 150°), indicating full controlled extension."
                )

            if angles["trunk_lean"] > 20.0:
                explanations.append(
                    f"Torso lean is {angles['trunk_lean']}° — you may be using momentum to lift the weight."
                )
                solutions.append(
                    "Stand tall with ribs stacked over hips (or sit against a bench) and lower the weight slightly if needed."
                )

        else:
            posture_summary = "Neutral Standing / General Exercise Posture"
            explanations.append(
                f"Detected upright alignment: knee {angles['knee_angle']}°, elbow {angles['elbow_angle']}°, trunk lean {angles['trunk_lean']}°."
            )
            solutions.append(
                "Select 'Squats' or 'Bicep Curls' above—or upload a frame mid-movement—for exercise-specific biomechanical coaching."
            )

        if not solutions:
            solutions.append(
                "Form looks solid in this frame! Keep controlled tempo and steady breathing throughout the set."
            )

        annotated = estimator.draw_skeleton(
            bgr_image.copy(),
            pose_frame,
            angles=angles,
            state_label=detected_exercise.upper(),
            rep_count=0,
        )

        return ImageAnalysisResult(
            annotated_bgr=annotated,
            detected=True,
            side_used=side,
            detected_exercise=detected_exercise,
            posture_summary=posture_summary,
            angles=angles,
            explanations=explanations,
            solutions=solutions,
        )
    finally:
        estimator.close()
