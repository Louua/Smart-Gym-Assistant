"""Evaluation metrics for Smart Gym Assistant.

Computes:
- Repetition count error (E = |N_predicted - N_real|) and Mean Absolute Error (MAE).
- State / phase detection classification metrics (Precision, Recall, F1).
- System performance metrics (Processing FPS, mean latency, p95 latency).
- Tracking robustness (Mean keypoint visibility, dropped frame ratio).
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np


def compute_rep_error(predicted_reps: int, ground_truth_reps: int) -> int:
    """Repetition Count Error: E = |N_predicted - N_real|."""
    return abs(int(predicted_reps) - int(ground_truth_reps))


def compute_phase_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-phase Precision, Recall, and F1 score against frame-level ground truth."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    metrics: Dict[str, Dict[str, float]] = {}

    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for yt in y_true if yt == label),
        }

    return metrics


def compute_performance_metrics(frame_latencies_ms: List[float]) -> Dict[str, float]:
    """Compute system latency and throughput statistics."""
    if not frame_latencies_ms:
        return {"mean_latency_ms": 0.0, "p95_latency_ms": 0.0, "fps": 0.0}

    mean_lat = float(np.mean(frame_latencies_ms))
    p95_lat = float(np.percentile(frame_latencies_ms, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps": round(fps, 1),
    }


def compute_robustness_metrics(
    visibilities: List[float],
    total_frames: int,
    dropped_frames: int,
) -> Dict[str, float]:
    """Compute pose detection robustness under visual noise or occlusion."""
    mean_vis = float(np.mean(visibilities)) if visibilities else 0.0
    drop_pct = (dropped_frames / total_frames * 100.0) if total_frames > 0 else 0.0

    return {
        "mean_keypoint_visibility": round(mean_vis, 3),
        "dropped_frames": dropped_frames,
        "dropped_frame_pct": round(drop_pct, 2),
    }
