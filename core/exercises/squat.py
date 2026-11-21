"""Squat exercise tracker using a 4-state Finite State Machine.

States:
    STANDING   -> Knee fully extended (>= standing_min, default 160)
    DESCENDING -> Knee angle dropping (< descent_start, default 150)
    BOTTOM     -> Depth threshold reached (<= bottom_max, default 100)
    ASCENDING  -> Knee angle increasing again (> bottom_max + hysteresis, default 110)

A repetition is valid only upon completion of:
    STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING
"""

from __future__ import annotations
import math
from typing import Dict, Any, List
from core.models import PoseFrame, FrameResult, RepEvent
from core.exercises.base import Exercise
from core.geometry import angle_between, denormalize_point, trunk_inclination, select_side
from core.feedback import format_feedback_list


class SquatExercise(Exercise):
    """Tracks and evaluates squat repetitions using biomechanical FSM."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name="Squats", config=config)

        squat_cfg = self.config.get("squat", {})
        self.standing_min = float(squat_cfg.get("standing_min", 160.0))
        self.descent_start = float(squat_cfg.get("descent_start", 150.0))
        self.bottom_max = float(squat_cfg.get("bottom_max", 100.0))
        self.ascent_hysteresis = float(squat_cfg.get("ascent_hysteresis", 10.0))
        self.trunk_lean_max = float(squat_cfg.get("trunk_lean_max", 45.0))
        self.count_incomplete_ext = bool(squat_cfg.get("count_incomplete_extension", False))

        self.current_state = "STANDING"
        self._prev_knee_angle = self.standing_min

    def reset(self) -> None:
        super().reset()
        self.current_state = "STANDING"
        self._prev_knee_angle = self.standing_min

    def update(
        self,
        frame: PoseFrame,
        image_width: int,
        image_height: int,
        side_override: str = "auto",
    ) -> FrameResult:
        t = frame.timestamp_s
        frame_angles: Dict[str, float] = {
            "knee": math.nan,
            "hip": math.nan,
            "ankle": math.nan,
            "trunk_inclination": math.nan,
        }
        active_issues: List[str] = []
        mean_vis = 0.0

        if not frame.keypoints:
            return FrameResult(
                frame_idx=frame.frame_idx,
                timestamp_s=t,
                angles=frame_angles,
                state=self.current_state,
                rep_count=self.rep_count,
                active_feedback=format_feedback_list(["low_visibility"]),
                mean_visibility=0.0,
            )

        side = select_side(frame, side_override)
        sh_name = f"{side}_shoulder"
        hip_name = f"{side}_hip"
        knee_name = f"{side}_knee"
        ank_name = f"{side}_ankle"

        kps = frame.keypoints
        req_joints = [sh_name, hip_name, knee_name, ank_name]
        visibilities = [kps[j].visibility for j in req_joints if j in kps]
        mean_vis = sum(visibilities) / len(visibilities) if visibilities else 0.0

        if mean_vis < 0.4:
            active_issues.append("low_visibility")

        if all(j in kps for j in req_joints):
            sh_pt = denormalize_point(kps[sh_name], image_width, image_height)
            hip_pt = denormalize_point(kps[hip_name], image_width, image_height)
            knee_pt = denormalize_point(kps[knee_name], image_width, image_height)
            ank_pt = denormalize_point(kps[ank_name], image_width, image_height)

            # Knee angle: Hip - Knee - Ankle
            knee_angle = angle_between(hip_pt, knee_pt, ank_pt)
            # Hip angle: Shoulder - Hip - Knee
            hip_angle = angle_between(sh_pt, hip_pt, knee_pt)
            # Ankle angle: Knee - Ankle - Foot projection
            # Approximate ankle angle: Knee - Ankle - Horizontal forward vector
            foot_proxy = (ank_pt[0] + (50 if side == "right" else -50), ank_pt[1])
            ankle_angle = angle_between(knee_pt, ank_pt, foot_proxy)
            # Trunk inclination
            trunk_lean = trunk_inclination(sh_pt, hip_pt)

            frame_angles["knee"] = knee_angle
            frame_angles["hip"] = hip_angle
            frame_angles["ankle"] = ankle_angle
            frame_angles["trunk_inclination"] = trunk_lean

            # Biomechanical form check: excessive trunk lean
            if not math.isnan(trunk_lean) and trunk_lean > self.trunk_lean_max:
                active_issues.append("excessive_trunk_lean")
                self._rep_issues.add("excessive_trunk_lean")

            if not math.isnan(knee_angle):
                self._advance_fsm(knee_angle, t, active_issues)
                self._prev_knee_angle = knee_angle

        return FrameResult(
            frame_idx=frame.frame_idx,
            timestamp_s=t,
            angles=frame_angles,
            state=self.current_state,
            rep_count=self.rep_count,
            active_feedback=format_feedback_list(active_issues),
            mean_visibility=mean_vis,
        )

    def _advance_fsm(self, knee: float, timestamp_s: float, active_issues: List[str]) -> None:
        """Execute state machine transitions with hysteresis and rejection logic."""
        if self.current_state == "STANDING":
            if knee < self.descent_start:
                # Begin new repetition
                self.current_state = "DESCENDING"
                self._rep_start_s = timestamp_s
                self._rep_min_angle = knee
                self._rep_max_angle = knee
                self._rep_issues.clear()
            else:
                self._rep_max_angle = max(self._rep_max_angle, knee)

        elif self.current_state == "DESCENDING":
            self._rep_min_angle = min(self._rep_min_angle, knee)
            self._rep_max_angle = max(self._rep_max_angle, knee)

            if knee <= self.bottom_max:
                # Sufficient depth achieved
                self.current_state = "BOTTOM"
            elif knee > self.descent_start:
                # Movement reversed before reaching bottom depth -> REJECT
                self.rejected_attempts += 1
                active_issues.append("insufficient_depth")
                self.reps.append(
                    RepEvent(
                        rep_index=len(self.reps) + 1,
                        start_s=self._rep_start_s,
                        end_s=timestamp_s,
                        min_angle=self._rep_min_angle,
                        max_angle=self._rep_max_angle,
                        valid=False,
                        issues=["insufficient_depth"],
                    )
                )
                self.current_state = "STANDING"

        elif self.current_state == "BOTTOM":
            self._rep_min_angle = min(self._rep_min_angle, knee)
            # Hysteresis: require knee angle to rise above bottom_max + ascent_hysteresis
            if knee > (self.bottom_max + self.ascent_hysteresis):
                self.current_state = "ASCENDING"

        elif self.current_state == "ASCENDING":
            self._rep_max_angle = max(self._rep_max_angle, knee)

            if knee >= self.standing_min:
                # Fully locked out standing position -> VALID REP
                self.reps.append(
                    RepEvent(
                        rep_index=len(self.reps) + 1,
                        start_s=self._rep_start_s,
                        end_s=timestamp_s,
                        min_angle=self._rep_min_angle,
                        max_angle=self._rep_max_angle,
                        valid=True,
                        issues=sorted(list(self._rep_issues)),
                    )
                )
                self.current_state = "STANDING"
                self._rep_issues.clear()
            elif knee < self._prev_knee_angle - 5.0 and knee < self.descent_start:
                # Ascending reversed downward without reaching standing lockout
                if not self.count_incomplete_ext:
                    self.rejected_attempts += 1
                    active_issues.append("incomplete_extension")
                    self.reps.append(
                        RepEvent(
                            rep_index=len(self.reps) + 1,
                            start_s=self._rep_start_s,
                            end_s=timestamp_s,
                            min_angle=self._rep_min_angle,
                            max_angle=self._rep_max_angle,
                            valid=False,
                            issues=["incomplete_extension"],
                        )
                    )
                    # Begin descending for the next attempt
                    self.current_state = "DESCENDING"
                    self._rep_start_s = timestamp_s
                    self._rep_min_angle = knee
                    self._rep_issues.clear()
