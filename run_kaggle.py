"""Headless batch runner for Kaggle and cloud environments.

Executes the Smart Gym Assistant pipeline on an input video, writes out an annotated MP4
video with skeletal HUD overlay, generates CSV/JSON session exports, and prints summary metrics.
No GUI or display server required.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import json
import cv2
import pandas as pd

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.exercises import create_exercise
from core.video_input import FileVideoSource
from core.pipeline import WorkoutPipeline
from core.config import load_config


def process_video_headless(
    video_path: str | Path,
    exercise_name: str = "Squats",
    side_override: str = "auto",
    output_video_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
) -> dict:
    """Process video headlessly and output annotated video and session metrics."""
    v_path = Path(video_path)
    if not v_path.exists():
        raise FileNotFoundError(f"Video file not found: {v_path}")

    cfg = load_config()
    source = FileVideoSource(v_path)
    exercise = create_exercise(exercise_name, config=cfg)
    pipeline = WorkoutPipeline(
        exercise=exercise,
        source=source,
        config=cfg,
        side_override=side_override,
    )

    writer = None
    if output_video_path is not None:
        out_v = Path(output_video_path)
        out_v.parent.mkdir(parents=True, exist_ok=True)
        w, h = source.dimensions
        # Use avc1 / H264 or mp4v for high compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_v), fourcc, source.fps, (w, h))

    print(f"--> Processing {v_path.name} ({exercise_name}, side={side_override})...")
    frame_count = 0

    for annotated_bgr, frame_res in pipeline.run():
        if writer is not None:
            writer.write(annotated_bgr)
        frame_count += 1
        if frame_count % 60 == 0:
            print(f"    Frame {frame_count} | Reps: {frame_res.rep_count} | State: {frame_res.state}")

    if writer is not None:
        writer.release()
        print(f"[OK] Annotated video saved to: {output_video_path}")

    source.release()
    session = pipeline.compute_session_result()

    valid_reps = len([r for r in session.reps if r.valid])
    print(f"\n==========================================")
    print(f"Session Complete: {session.exercise}")
    print(f"Valid Repetitions: {valid_reps}")
    print(f"Rejected Attempts: {session.rejected_attempts}")
    print(f"Overall Form Score: {session.summary.get('overall_form', 0.0):.1f}%")
    print(f"Processing Throughput: {session.summary.get('effective_fps', 0.0):.1f} FPS")
    print(f"==========================================\n")

    # Export JSON
    if output_json_path is not None:
        out_j = Path(output_json_path)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as f:
            f.write(session.to_json(indent=2))
        print(f"[OK] Session JSON saved to: {output_json_path}")

    # Export CSV of Reps
    if output_csv_path is not None:
        out_c = Path(output_csv_path)
        out_c.parent.mkdir(parents=True, exist_ok=True)
        rep_rows = []
        for r in session.reps:
            rep_rows.append({
                "rep_index": r.rep_index,
                "valid": r.valid,
                "start_s": round(r.start_s, 2),
                "end_s": round(r.end_s, 2),
                "duration_s": round(r.end_s - r.start_s, 2),
                "min_angle": round(r.min_angle, 1),
                "max_angle": round(r.max_angle, 1),
                "rom": round(r.max_angle - r.min_angle, 1),
                "issues": "; ".join(r.issues) if r.issues else "None",
            })
        df_reps = pd.DataFrame(rep_rows)
        df_reps.to_csv(out_c, index=False)
        print(f"[OK] Repetition CSV saved to: {output_csv_path}")

    return session.to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless workout tracker for Kaggle/Cloud")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--exercise", type=str, choices=["Squats", "Bicep Curls"], default="Squats")
    parser.add_argument("--side", type=str, choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--out_video", type=str, default="output_annotated.mp4")
    parser.add_argument("--out_json", type=str, default="session.json")
    parser.add_argument("--out_csv", type=str, default="reps.csv")
    args = parser.parse_args()

    process_video_headless(
        video_path=args.video,
        exercise_name=args.exercise,
        side_override=args.side,
        output_video_path=args.out_video,
        output_json_path=args.out_json,
        output_csv_path=args.out_csv,
    )
