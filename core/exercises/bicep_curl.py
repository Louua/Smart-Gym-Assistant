"""Bicep Curl exercise tracker using a 4-state Finite State Machine.

States:
    EXTENDED -> Arm fully extended (elbow >= extended_min, default 150)
    CURLING  -> Elbow angle decreasing (< curling_start, default 140)
    FLEXED   -> Peak flexion reached (elbow <= flexed_max, default 50)
    LOWERING -> Elbow angle increasing again (> lowering_start, default 60)

Monitors shoulder stability: detects shoulder displacement cheat/swinging.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple
from core.models import PoseFrame, FrameResult, RepEvent
from core.exercises.base import Exercise
from core.geometry import angle_between, denormalize_point, select_side
from core.feedback import format_feedback_list


class BicepCurlExercise(Exercise):
    """Tracks and evaluates bicep curl repetitions using biomechanical FSM."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name="Bicep Curls", config=config)

        curl_cfg = self.config.get("bicep_curl", {})
        self.extended_min = float(curl_cfg.get("extended_min", 150.0))
        self.curling_start = float(curl_cfg.get("curling_start", 140.0))
        self.flexed_max = float(curl_cfg.get("flexed_max", 50.0))
        self.lowering_start = float(curl_cfg.get("lowering_start", 60.0))
        self.shoulder_drift_max = float(curl_cfg.get("shoulder_drift_max", 0.08))

        self.current_state = "EXTENDED"
        self._prev_elbow_angle = self.extended_min
        self._initial_shoulder_pos: Tuple[float, float] | None = None
        self._torso_length: float = 1.0

    def reset(self) -> None:
        super().reset()
        self.current_state = "EXTENDED"
        self._prev_elbow_angle = self.extended_min
        self._initial_shoulder_pos = None
        self._torso_length = 1.0

    def update(
        self,
        frame: PoseFrame,
        image_width: int,
        image_height: int,
        side_override: str = "auto",
    ) -> FrameResult:
        t = frame.timestamp_s
        frame_angles: Dict[str, float] = {
            "elbow": math.nan,
            "shoulder": math.nan,
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
        el_name = f"{side}_elbow"
        wr_name = f"{side}_wrist"
        hip_name = f"{side}_hip"

        kps = frame.keypoints
        req_joints = [sh_name, el_name, wr_name, hip_name]
        visibilities = [kps[j].visibility for j in req_joints if j in kps]
        mean_vis = sum(visibilities) / len(visibilities) if visibilities else 0.0

        if mean_vis < 0.4:
            active_issues.append("low_visibility")

        if all(j in kps for j in req_joints):
            sh_pt = denormalize_point(kps[sh_name], image_width, image_height)
            el_pt = denormalize_point(kps[el_name], image_width, image_height)
            wr_pt = denormalize_point(kps[wr_name], image_width, image_height)
            hip_pt = denormalize_point(kps[hip_name], image_width, image_height)

            # Elbow angle: Shoulder - Elbow - Wrist
            elbow_angle = angle_between(sh_pt, el_pt, wr_pt)
            # Shoulder angle: Hip - Shoulder - Elbow
            shoulder_angle = angle_between(hip_pt, sh_pt, el_pt)

            frame_angles["elbow"] = elbow_angle
            frame_angles["shoulder"] = shoulder_angle

            # Torso length for normalized drift calculation
            dx = sh_pt[0] - hip_pt[0]
            dy = sh_pt[1] - hip_pt[1]
            torso_len = math.sqrt(dx * dx + dy * dy)
            if torso_len > 10.0:
                self._torso_length = torso_len

            # Check shoulder drift/compensation
            if self._initial_shoulder_pos is not None:
                sx_diff = sh_pt[0] - self._initial_shoulder_pos[0]
                sy_diff = sh_pt[1] - self._initial_shoulder_pos[1]
                drift_dist = math.sqrt(sx_diff * sx_diff + sy_diff * sy_diff)
                normalized_drift = drift_dist / self._torso_length
                if normalized_drift > self.shoulder_drift_max:
                    active_issues.append("shoulder_compensation")
                    self._rep_issues.add("shoulder_compensation")

            if not math.isnan(elbow_angle):
                self._advance_fsm(elbow_angle, sh_pt, t, active_issues)
                self._prev_elbow_angle = elbow_angle

        return FrameResult(
            frame_idx=frame.frame_idx,
            timestamp_s=t,
            angles=frame_angles,
            state=self.current_state,
            rep_count=self.rep_count,
            active_feedback=format_feedback_list(active_issues),
            mean_visibility=mean_vis,
        )

    def _advance_fsm(
        self,
        elbow: float,
        shoulder_pt: Tuple[float, float],
        timestamp_s: float,
        active_issues: List[str],
    ) -> None:
        """Execute state machine transitions with hysteresis and rejection logic."""
        if self.current_state == "EXTENDED":
            if elbow < self.curling_start:
                # Repetition starts
                self.current_state = "CURLING"
                self._rep_start_s = timestamp_s
                self._rep_min_angle = elbow
                self._rep_max_angle = elbow
                self._rep_issues.clear()
                self._initial_shoulder_pos = shoulder_pt
            else:
                self._rep_max_angle = max(self._rep_max_angle, elbow)

        elif self.current_state == "CURLING":
            self._rep_min_angle = min(self._rep_min_angle, elbow)
            self._rep_max_angle = max(self._rep_max_angle, elbow)

            if elbow <= self.flexed_max:
                # Full contraction reached
                self.current_state = "FLEXED"
            elif elbow > self.curling_start:
                # Reversed back before full flexion -> REJECT
                self.rejected_attempts += 1
                active_issues.append("incomplete_flexion")
                self.reps.append(
                    RepEvent(
                        rep_index=len(self.reps) + 1,
                        start_s=self._rep_start_s,
                        end_s=timestamp_s,
                        min_angle=self._rep_min_angle,
                        max_angle=self._rep_max_angle,
                        valid=False,
                        issues=["incomplete_flexion"],
                    )
                )
                self.current_state = "EXTENDED"
                self._initial_shoulder_pos = None

        elif self.current_state == "FLEXED":
            self._rep_min_angle = min(self._rep_min_angle, elbow)
            # Hysteresis: angle must open past lowering_start
            if elbow > self.lowering_start:
                self.current_state = "LOWERING"

        elif self.current_state == "LOWERING":
            self._rep_max_angle = max(self._rep_max_angle, elbow)

            if elbow >= self.extended_min:
                # Reached full extension -> VALID REP
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
                self.current_state = "EXTENDED"
                self._initial_shoulder_pos = None
                self._rep_issues.clear()
            elif elbow < self._prev_elbow_angle - 5.0 and elbow < self.curling_start:
                # User began another curl before fully extending
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
                self.current_state = "CURLING"
                self._rep_start_s = timestamp_s
                self._rep_min_angle = elbow
                self._rep_issues.clear()
                self._initial_shoulder_pos = shoulder_pt
