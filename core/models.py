"""Data contracts and model definitions for Smart Gym Assistant.

Strictly decoupled data models used across all pipeline stages.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass(frozen=True)
class Keypoint:
    """Normalized keypoint coordinate with detection visibility."""
    x: float            # normalized [0, 1]
    y: float            # normalized [0, 1]
    visibility: float   # MediaPipe confidence [0, 1]


@dataclass
class PoseFrame:
    """Estimated pose coordinates for a single video frame."""
    frame_idx: int
    timestamp_s: float
    keypoints: dict[str, Keypoint] | None   # None if no person detected
    # Required keys when detected:
    # 'left_shoulder', 'right_shoulder',
    # 'left_elbow', 'right_elbow',
    # 'left_wrist', 'right_wrist',
    # 'left_hip', 'right_hip',
    # 'left_knee', 'right_knee',
    # 'left_ankle', 'right_ankle'


@dataclass
class FrameResult:
    """Biomechanical assessment result for a single frame."""
    frame_idx: int
    timestamp_s: float
    angles: dict[str, float]        # e.g. knee, hip, ankle, trunk_inclination, elbow
    state: str                      # Current FSM state
    rep_count: int
    active_feedback: list[str] = field(default_factory=list)
    mean_visibility: float = 0.0


@dataclass
class RepEvent:
    """Summary of an individual repetition attempt."""
    rep_index: int
    start_s: float
    end_s: float
    min_angle: float                # Peak flexion reached (squat knee / curl elbow)
    max_angle: float                # Peak extension reached
    valid: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class SessionResult:
    """Complete summary of a processed workout session."""
    exercise: str
    source: str
    fps: float
    frames: list[FrameResult]
    reps: list[RepEvent]
    rejected_attempts: int
    summary: dict[str, float]       # form_consistency, range_of_motion, stability, overall_form (0-100)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to serializable dictionary."""
        return {
            "exercise": self.exercise,
            "source": self.source,
            "fps": self.fps,
            "rep_count": len([r for r in self.reps if r.valid]),
            "rejected_attempts": self.rejected_attempts,
            "summary": self.summary,
            "reps": [asdict(r) for r in self.reps],
            "frame_count": len(self.frames),
        }

    def to_json(self, indent: int = 2) -> str:
        """Export session result as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
