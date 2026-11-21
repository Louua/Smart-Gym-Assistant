"""Synthetic stick-figure video generator for smoke testing and automated validation.

Generates realistic 2D human stick figure animations of squats and bicep curls.
Useful for CI/CD, unit integration, and verifying the pipeline without a camera or manual video file.
"""

from __future__ import annotations
import math
import argparse
from pathlib import Path
import cv2
import numpy as np


def generate_squat_video(
    output_path: str | Path,
    num_reps: int = 4,
    include_shallow: bool = True,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> None:
    """Generate a synthetic squat video.

    Draws a side-view human stick figure performing squats with smooth sinusoidal articulation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Base coordinates for side view
    ankle_fixed = (int(width * 0.45), int(height * 0.82))
    femur_len = int(height * 0.22)
    tibia_len = int(height * 0.22)
    torso_len = int(height * 0.25)
    neck_len = int(height * 0.05)
    head_rad = int(height * 0.045)
    arm_len = int(height * 0.22)

    total_patterns = num_reps + (1 if include_shallow else 0)
    rep_duration_s = 3.0
    frames_per_rep = int(fps * rep_duration_s)

    # Initial standing pause
    for _ in range(fps):
        frame = _render_squat_frame(
            knee_angle_deg=170.0,
            ankle_fixed=ankle_fixed,
            femur_len=femur_len,
            tibia_len=tibia_len,
            torso_len=torso_len,
            neck_len=neck_len,
            head_rad=head_rad,
            arm_len=arm_len,
            width=width,
            height=height,
        )
        out.write(frame)

    for r in range(total_patterns):
        is_shallow = include_shallow and (r == 1)  # Make rep 1 a shallow attempt
        min_angle = 125.0 if is_shallow else 90.0

        for f in range(frames_per_rep):
            phase = (f / frames_per_rep) * 2.0 * math.pi
            # Sinusoid: starts at 170, dips to min_angle, returns to 170
            angle = min_angle + (170.0 - min_angle) * (0.5 * (1.0 + math.cos(phase)))

            frame = _render_squat_frame(
                knee_angle_deg=angle,
                ankle_fixed=ankle_fixed,
                femur_len=femur_len,
                tibia_len=tibia_len,
                torso_len=torso_len,
                neck_len=neck_len,
                head_rad=head_rad,
                arm_len=arm_len,
                width=width,
                height=height,
            )
            out.write(frame)

    # Final pause
    for _ in range(fps):
        frame = _render_squat_frame(
            knee_angle_deg=170.0,
            ankle_fixed=ankle_fixed,
            femur_len=femur_len,
            tibia_len=tibia_len,
            torso_len=torso_len,
            neck_len=neck_len,
            head_rad=head_rad,
            arm_len=arm_len,
            width=width,
            height=height,
        )
        out.write(frame)

    out.release()


def _render_squat_frame(
    knee_angle_deg: float,
    ankle_fixed: tuple[int, int],
    femur_len: int,
    tibia_len: int,
    torso_len: int,
    neck_len: int,
    head_rad: int,
    arm_len: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Render single stick figure squat frame."""
    frame = np.full((height, width, 3), 30, dtype=np.uint8)

    # Floor line
    cv2.line(frame, (0, ankle_fixed[1] + 15), (width, ankle_fixed[1] + 15), (70, 70, 70), 2)

    # Foot
    foot_end = (ankle_fixed[0] + 45, ankle_fixed[1])
    cv2.line(frame, ankle_fixed, foot_end, (220, 220, 220), 4)

    # Tibia angle leans slightly forward as knee bends
    bend_ratio = (170.0 - knee_angle_deg) / 80.0
    tibia_lean_rad = math.radians(10.0 + 20.0 * bend_ratio)

    knee_x = int(ankle_fixed[0] + tibia_len * math.sin(tibia_lean_rad))
    knee_y = int(ankle_fixed[1] - tibia_len * math.cos(tibia_lean_rad))
    knee_pt = (knee_x, knee_y)

    # Hip position: knee angle determines hip-femur angle
    knee_rad = math.radians(knee_angle_deg)
    femur_angle = tibia_lean_rad - (math.pi - knee_rad)

    hip_x = int(knee_x - femur_len * math.sin(femur_angle))
    hip_y = int(knee_y - femur_len * math.cos(femur_angle))
    hip_pt = (hip_x, hip_y)

    # Torso: leans forward counterbalancing hip
    torso_lean_rad = math.radians(15.0 + 25.0 * bend_ratio)
    shoulder_x = int(hip_x + torso_len * math.sin(torso_lean_rad))
    shoulder_y = int(hip_y - torso_len * math.cos(torso_lean_rad))
    shoulder_pt = (shoulder_x, shoulder_y)

    # Head
    head_center = (
        int(shoulder_x + (neck_len + head_rad) * math.sin(torso_lean_rad)),
        int(shoulder_y - (neck_len + head_rad) * math.cos(torso_lean_rad)),
    )

    # Arm reaching forward for squat balance
    wrist_pt = (shoulder_x + arm_len, shoulder_y + 10)

    # Draw lines
    line_color = (240, 240, 240)
    thickness = 5

    cv2.line(frame, ankle_fixed, knee_pt, line_color, thickness)
    cv2.line(frame, knee_pt, hip_pt, line_color, thickness)
    cv2.line(frame, hip_pt, shoulder_pt, line_color, thickness)
    cv2.line(frame, shoulder_pt, wrist_pt, line_color, thickness)
    cv2.circle(frame, head_center, head_rad, line_color, -1)

    # Joint nodes
    for pt in [ankle_fixed, knee_pt, hip_pt, shoulder_pt, wrist_pt]:
        cv2.circle(frame, pt, 7, (0, 180, 255), -1)

    return frame


def generate_curl_video(
    output_path: str | Path,
    num_reps: int = 4,
    include_incomplete: bool = True,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> None:
    """Generate synthetic bicep curl video."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Base coords
    sh_pt = (int(width * 0.5), int(height * 0.3))
    hip_pt = (int(width * 0.5), int(height * 0.65))
    knee_pt = (int(width * 0.5), int(height * 0.85))
    ankle_pt = (int(width * 0.5), int(height * 0.95))
    upper_arm_len = int(height * 0.18)
    forearm_len = int(height * 0.18)

    elbow_pt = (sh_pt[0], sh_pt[1] + upper_arm_len)

    total_patterns = num_reps + (1 if include_incomplete else 0)
    rep_duration_s = 3.0
    frames_per_rep = int(fps * rep_duration_s)

    # Initial extended pause
    for _ in range(fps):
        frame = _render_curl_frame(
            elbow_angle_deg=160.0,
            sh_pt=sh_pt,
            elbow_pt=elbow_pt,
            forearm_len=forearm_len,
            hip_pt=hip_pt,
            knee_pt=knee_pt,
            ankle_pt=ankle_pt,
            width=width,
            height=height,
        )
        out.write(frame)

    for r in range(total_patterns):
        is_partial = include_incomplete and (r == 1)
        min_angle = 85.0 if is_partial else 40.0

        for f in range(frames_per_rep):
            phase = (f / frames_per_rep) * 2.0 * math.pi
            angle = min_angle + (160.0 - min_angle) * (0.5 * (1.0 + math.cos(phase)))

            frame = _render_curl_frame(
                elbow_angle_deg=angle,
                sh_pt=sh_pt,
                elbow_pt=elbow_pt,
                forearm_len=forearm_len,
                hip_pt=hip_pt,
                knee_pt=knee_pt,
                ankle_pt=ankle_pt,
                width=width,
                height=height,
            )
            out.write(frame)

    # Final pause
    for _ in range(fps):
        frame = _render_curl_frame(
            elbow_angle_deg=160.0,
            sh_pt=sh_pt,
            elbow_pt=elbow_pt,
            forearm_len=forearm_len,
            hip_pt=hip_pt,
            knee_pt=knee_pt,
            ankle_pt=ankle_pt,
            width=width,
            height=height,
        )
        out.write(frame)

    out.release()


def _render_curl_frame(
    elbow_angle_deg: float,
    sh_pt: tuple[int, int],
    elbow_pt: tuple[int, int],
    forearm_len: int,
    hip_pt: tuple[int, int],
    knee_pt: tuple[int, int],
    ankle_pt: tuple[int, int],
    width: int,
    height: int,
) -> np.ndarray:
    """Render single stick figure curl frame."""
    frame = np.full((height, width, 3), 30, dtype=np.uint8)

    # Forearm rotation around elbow
    rad = math.radians(elbow_angle_deg)
    wrist_x = int(elbow_pt[0] + forearm_len * math.sin(math.pi - rad))
    wrist_y = int(elbow_pt[1] + forearm_len * math.cos(math.pi - rad))
    wrist_pt = (wrist_x, wrist_y)

    # Head
    head_center = (sh_pt[0], sh_pt[1] - 35)

    line_color = (240, 240, 240)
    thickness = 5

    # Torso & legs
    cv2.line(frame, sh_pt, hip_pt, line_color, thickness)
    cv2.line(frame, hip_pt, knee_pt, line_color, thickness)
    cv2.line(frame, knee_pt, ankle_pt, line_color, thickness)
    cv2.circle(frame, head_center, 22, line_color, -1)

    # Arm
    cv2.line(frame, sh_pt, elbow_pt, line_color, thickness)
    cv2.line(frame, elbow_pt, wrist_pt, (0, 255, 0), thickness)

    # Joints
    for pt in [sh_pt, hip_pt, knee_pt, ankle_pt, elbow_pt, wrist_pt]:
        cv2.circle(frame, pt, 6, (0, 180, 255), -1)

    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic workout videos")
    parser.add_argument("--exercise", choices=["squat", "curl"], default="squat")
    parser.add_argument("--out", type=str, default="data/videos/demo.mp4")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    if args.exercise == "squat":
        generate_squat_video(args.out, num_reps=args.reps)
        print(f"Generated synthetic squat video: {args.out}")
    else:
        generate_curl_video(args.out, num_reps=args.reps)
        print(f"Generated synthetic curl video: {args.out}")
