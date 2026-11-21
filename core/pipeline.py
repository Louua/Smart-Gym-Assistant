"""Pipeline orchestrator linking video input, pose estimation, tracking, and exercise analysis.

Supports both interactive real-time frame streaming and batch execution.
Computes session summary scores (0-100): Form Consistency, Range of Motion, Stability, Overall Form.
"""

from __future__ import annotations
import time
import math
from typing import Generator, Tuple, Dict, Any, List
import numpy as np

from core.models import FrameResult, SessionResult, RepEvent
from core.video_input import VideoSource
from core.pose_estimator import PoseEstimator
from core.tracker import TemporalTracker
from core.exercises.base import Exercise
from core.config import load_config


class WorkoutPipeline:
    """Orchestrates end-to-end processing across Modules 1 to 4."""

    def __init__(
        self,
        exercise: Exercise,
        source: VideoSource,
        config: Dict[str, Any] | None = None,
        pose_estimator: PoseEstimator | None = None,
        tracker: TemporalTracker | None = None,
        side_override: str = "auto",
    ) -> None:
        self.exercise = exercise
        self.source = source
        self.config = config or load_config()
        self.side_override = side_override

        # Initialize or reuse pose estimator
        if pose_estimator is not None:
            self.pose_estimator = pose_estimator
            self._owns_pose_estimator = False
        else:
            self.pose_estimator = PoseEstimator(
                model_complexity=1,
                min_detection_confidence=float(self.config.get("visibility", {}).get("min_visibility", 0.5)),
                min_tracking_confidence=float(self.config.get("visibility", {}).get("min_visibility", 0.5)),
            )
            self._owns_pose_estimator = True

        # Initialize or reuse tracker
        if tracker is not None:
            self.tracker = tracker
        else:
            smooth_cfg = self.config.get("smoothing", {})
            vis_cfg = self.config.get("visibility", {})
            self.tracker = TemporalTracker(
                alpha=float(smooth_cfg.get("alpha", 0.4)),
                smoothing_method=str(smooth_cfg.get("method", "ema")),
                window_size=int(smooth_cfg.get("window", 7)),
                min_visibility=float(vis_cfg.get("min_visibility", 0.5)),
                max_hold_frames=int(vis_cfg.get("max_hold_frames", 10)),
            )

        self.frame_latencies: List[float] = []
        self.frame_results: List[FrameResult] = []

    def run(self) -> Generator[Tuple[np.ndarray, FrameResult], None, None]:
        """Generator yielding (annotated_bgr_frame, FrameResult) for each processed frame."""
        self.exercise.reset()
        self.tracker.reset()
        self.frame_latencies.clear()
        self.frame_results.clear()

        w, h = self.source.dimensions

        for frame_idx, timestamp_s, bgr_image in self.source.frames():
            t0 = time.perf_counter()

            # Step 1 & 2: Pose estimation
            raw_pose = self.pose_estimator.process_frame(bgr_image, frame_idx, timestamp_s)

            # Step 3: Temporal tracking & smoothing
            tracked_pose = self.tracker.update(raw_pose)

            # Step 4: Exercise analysis & FSM
            result = self.exercise.update(
                tracked_pose,
                image_width=w,
                image_height=h,
                side_override=self.side_override,
            )

            # Measure inference & analysis latency
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.frame_latencies.append(latency_ms)
            self.frame_results.append(result)

            # Draw HUD skeleton
            annotated = self.pose_estimator.draw_skeleton(
                bgr_image=bgr_image,
                pose_frame=tracked_pose,
                active_side=self.side_override if self.side_override in ("left", "right") else None,
                angles=result.angles,
                state=result.state,
                rep_count=result.rep_count,
                feedback=result.active_feedback,
            )

            yield (annotated, result)

        if self._owns_pose_estimator:
            self.pose_estimator.close()

    def run_to_completion(self) -> SessionResult:
        """Execute entire video and return aggregated SessionResult."""
        for _ in self.run():
            pass

        return self.compute_session_result()

    def compute_session_result(self) -> SessionResult:
        """Calculate aggregate scores and assemble SessionResult."""
        reps = list(self.exercise.reps)
        rejected = self.exercise.rejected_attempts
        valid_reps = [r for r in reps if r.valid]

        # 1. Form Consistency (0-100): Lower std deviation of depth and extension across reps
        if len(valid_reps) >= 2:
            depths = [r.min_angle for r in valid_reps]
            extensions = [r.max_angle for r in valid_reps]
            depth_std = float(np.std(depths))
            ext_std = float(np.std(extensions))
            avg_std = (depth_std + ext_std) / 2.0
            # std of 0 -> 100 score, std >= 20 -> 0 score
            consistency_score = max(0.0, min(100.0, 100.0 - (avg_std * 5.0)))
        elif len(valid_reps) == 1:
            consistency_score = 80.0
        else:
            consistency_score = 0.0

        # 2. Range of Motion (ROM) Score (0-100)
        # Squat target ROM: ~60-70 deg (160 down to 95); Curl target ROM: ~100 deg (150 down to 50)
        target_rom = 60.0 if "squat" in self.exercise.name.lower() else 95.0
        if valid_reps:
            roms = [r.max_angle - r.min_angle for r in valid_reps]
            mean_rom = float(np.mean(roms))
            rom_score = max(0.0, min(100.0, (mean_rom / target_rom) * 100.0))
        else:
            rom_score = 0.0

        # 3. Stability Score (0-100)
        # Measure penalty for issues detected during valid reps
        total_issues_count = sum(len(r.issues) for r in valid_reps) + (rejected * 2)
        penalty = total_issues_count * 10.0
        stability_score = max(0.0, min(100.0, 100.0 - penalty))

        # 4. Overall Form Score (0-100): Weighted sum
        weights = self.config.get("scoring", {}).get(
            "weights", {"consistency": 0.3, "rom": 0.4, "stability": 0.3}
        )
        w_c = float(weights.get("consistency", 0.3))
        w_r = float(weights.get("rom", 0.4))
        w_s = float(weights.get("stability", 0.3))
        total_w = w_c + w_r + w_s if (w_c + w_r + w_s) > 0 else 1.0

        overall_score = (
            w_c * consistency_score + w_r * rom_score + w_s * stability_score
        ) / total_w

        summary_metrics = {
            "form_consistency": round(consistency_score, 1),
            "range_of_motion": round(rom_score, 1),
            "stability": round(stability_score, 1),
            "overall_form": round(overall_score, 1),
            "mean_latency_ms": round(float(np.mean(self.frame_latencies)), 1) if self.frame_latencies else 0.0,
            "p95_latency_ms": round(float(np.percentile(self.frame_latencies, 95)), 1) if self.frame_latencies else 0.0,
            "effective_fps": round(1000.0 / np.mean(self.frame_latencies), 1) if self.frame_latencies and np.mean(self.frame_latencies) > 0 else 0.0,
        }

        return SessionResult(
            exercise=self.exercise.name,
            source=getattr(self.source, "file_path", "webcam"),
            fps=self.source.fps,
            frames=list(self.frame_results),
            reps=reps,
            rejected_attempts=rejected,
            summary=summary_metrics,
        )
