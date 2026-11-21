"""CLI Evaluation Harness for Smart Gym Assistant.

Runs workout pipeline across a benchmark dataset of videos with ground-truth labels.
Computes Repetition Count Error, MAE, phase F1, and system latency grouped by environmental conditions.
Outputs results to console, CSV, and formatted Markdown report.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import csv
import json
import pandas as pd
import numpy as np

from core.exercises import create_exercise
from core.video_input import FileVideoSource
from core.pipeline import WorkoutPipeline
from core.config import load_config
from evaluation.metrics import (
    compute_rep_error,
    compute_phase_classification_metrics,
    compute_performance_metrics,
    compute_robustness_metrics,
)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as Markdown table without external tabulate dependency."""
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = []
    for _, row in df.iterrows():
        cells = [str(row[col]) for col in df.columns]
        data_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_row, sep_row] + data_rows)


def run_evaluation(
    gt_csv_path: str | Path,
    video_dir: str | Path,
    output_report_path: str | Path,
) -> None:
    """Run batch evaluation across all videos in ground truth index."""
    gt_path = Path(gt_csv_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    v_dir = Path(video_dir)
    cfg = load_config()

    df_gt = pd.read_csv(gt_path)
    print(f"\n=======================================================")
    print(f"Smart Gym Assistant — Benchmark Evaluation Suite")
    print(f"Loaded {len(df_gt)} test cases from {gt_path}")
    print(f"=======================================================\n")

    results = []

    for idx, row in df_gt.iterrows():
        vid_filename = str(row["video"]).strip()
        vid_path = v_dir / vid_filename
        if not vid_path.exists():
            # Check relative to gt path
            vid_path = gt_path.parent / vid_filename
            if not vid_path.exists():
                print(f"[SKIP] Video file not found: {vid_filename}")
                continue

        exercise_type = str(row["exercise"]).strip()
        real_reps = int(row["real_reps"])
        lighting = str(row.get("lighting", "normal")).strip()
        camera_angle = str(row.get("camera_angle", "profile")).strip()
        occlusion = str(row.get("occlusion", "none")).strip()

        print(f"[{idx+1}/{len(df_gt)}] Processing {vid_filename} ({exercise_type})...")

        exercise = create_exercise(exercise_type, config=cfg)
        source = FileVideoSource(vid_path)
        pipeline = WorkoutPipeline(exercise=exercise, source=source, config=cfg)

        session = pipeline.run_to_completion()
        source.release()

        pred_reps = len([r for r in session.reps if r.valid])
        rep_err = compute_rep_error(pred_reps, real_reps)

        vis_list = [f.mean_visibility for f in session.frames]
        perf = compute_performance_metrics(pipeline.frame_latencies)
        robustness = compute_robustness_metrics(
            vis_list, len(session.frames), pipeline.tracker.dropped_frames
        )

        # Check for optional phase labels file
        phase_metrics = None
        phase_label_file = row.get("phase_labels_file")
        if pd.notna(phase_label_file) and str(phase_label_file).strip():
            phase_p = Path(str(phase_label_file).strip())
            if not phase_p.exists():
                phase_p = gt_path.parent / str(phase_label_file).strip()
            if phase_p.exists():
                df_phase = pd.read_csv(phase_p)
                true_phases = df_phase["phase"].tolist()[:len(session.frames)]
                pred_phases = [f.state for f in session.frames[:len(true_phases)]]
                phase_metrics = compute_phase_classification_metrics(true_phases, pred_phases)

        rec = {
            "video": vid_filename,
            "exercise": exercise_type,
            "lighting": lighting,
            "camera_angle": camera_angle,
            "occlusion": occlusion,
            "real_reps": real_reps,
            "predicted_reps": pred_reps,
            "rep_error": rep_err,
            "rejected_attempts": session.rejected_attempts,
            "mean_latency_ms": perf["mean_latency_ms"],
            "p95_latency_ms": perf["p95_latency_ms"],
            "fps": perf["fps"],
            "mean_visibility": robustness["mean_keypoint_visibility"],
            "dropped_frame_pct": robustness["dropped_frame_pct"],
            "form_consistency": session.summary.get("form_consistency", 0.0),
            "rom_score": session.summary.get("range_of_motion", 0.0),
            "stability_score": session.summary.get("stability", 0.0),
            "overall_form": session.summary.get("overall_form", 0.0),
        }
        results.append(rec)

    if not results:
        print("No evaluation runs completed. Check dataset paths.")
        return

    df_res = pd.DataFrame(results)

    # Save CSV
    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_p, index=False)
    print(f"\n[OK] Raw results saved to: {out_p}")

    # Generate Markdown Summary Report
    mae_overall = df_res["rep_error"].mean()
    mean_fps = df_res["fps"].mean()
    mean_vis = df_res["mean_visibility"].mean()

    md_report_path = out_p.with_suffix(".md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Smart Gym Assistant — Evaluation Report\n\n")
        f.write("## Overall Benchmark Summary\n\n")
        f.write(f"- **Total Videos Evaluated:** {len(df_res)}\n")
        f.write(f"- **Mean Absolute Repetition Error (MAE):** {mae_overall:.2f} reps\n")
        f.write(f"- **Average Processing Speed:** {mean_fps:.1f} FPS\n")
        f.write(f"- **Average Landmark Visibility:** {mean_vis:.3f}\n\n")

        f.write("## Repetition Error by Testing Condition\n\n")

        for cond in ["lighting", "camera_angle", "occlusion"]:
            f.write(f"### Grouped by `{cond}`\n\n")
            grp = df_res.groupby(cond).agg(
                count=("rep_error", "count"),
                mae=("rep_error", "mean"),
                avg_fps=("fps", "mean"),
                avg_vis=("mean_visibility", "mean"),
            ).reset_index()
            f.write(_df_to_markdown(grp))
            f.write("\n\n")

        f.write("## Detailed Video Results\n\n")
        cols_to_show = ["video", "exercise", "lighting", "camera_angle", "real_reps", "predicted_reps", "rep_error", "fps", "overall_form"]
        f.write(_df_to_markdown(df_res[cols_to_show]))
        f.write("\n")

    print(f"[OK] Markdown report saved to: {md_report_path}")
    print("\n--- Summary Metrics ---")
    print(f"Overall MAE: {mae_overall:.2f} reps")
    print(f"Average FPS: {mean_fps:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Smart Gym Assistant on annotated videos")
    parser.add_argument("--gt", type=str, default="evaluation/ground_truth_template.csv", help="Path to ground truth CSV")
    parser.add_argument("--videos", type=str, default="data/videos", help="Directory containing test videos")
    parser.add_argument("--out", type=str, default="evaluation/report.csv", help="Path to output results CSV")
    args = parser.parse_args()

    run_evaluation(args.gt, args.videos, args.out)
