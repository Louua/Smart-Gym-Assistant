"""Module: Personalized Fitness Recommendations Engine.

Generates safe, accessible, health-focused exercise plans adapted to the user's
personal metrics, fitness goal, experience level, equipment, and joint-friendly preferences.
Prioritizes safety, accessibility, and overall health rather than body-weight or appearance targets.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json


VALID_GOALS = (
    "Improving Fitness",
    "Building Strength",
    "Increasing Endurance",
    "Staying Active",
)

VALID_LEVELS = ("Beginner", "Intermediate", "Advanced")

VALID_EQUIPMENT = ("Bodyweight Only", "Dumbbells & Bands", "Full Gym")


@dataclass
class UserProfile:
    """User input profile for personalized fitness recommendations."""

    weight_kg: float
    height_cm: float
    gender: str
    goal: str = "Improving Fitness"
    experience_level: str = "Beginner"
    days_per_week: int = 3
    session_minutes: int = 30
    equipment: str = "Bodyweight Only"
    accessibility_notes: List[str] = field(default_factory=list)


@dataclass
class ExerciseItem:
    """Single recommended exercise with coaching and safety guidance."""

    name: str
    sets: int
    reps_or_duration: str
    rest_s: int
    coaching_cue: str
    safety_modification: str
    cv_supported: bool = False


@dataclass
class WorkoutDay:
    """One day of the weekly recommendation schedule."""

    day_label: str
    focus: str
    warmup: List[str]
    exercises: List[ExerciseItem]
    cooldown: List[str]


@dataclass
class RecommendationPlan:
    """Complete personalized exercise recommendation plan."""

    profile_summary: Dict[str, Any]
    safety_principles: List[str]
    weekly_schedule: List[WorkoutDay]
    progression_guidance: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# Personalized Exercise Recommendation Plan",
            "",
            f"- **Goal:** {self.profile_summary['goal']}",
            f"- **Experience Level:** {self.profile_summary['experience_level']}",
            f"- **Schedule:** {self.profile_summary['days_per_week']} days/week • ~{self.profile_summary['session_minutes']} min/session",
            f"- **Equipment:** {self.profile_summary['equipment']}",
            "",
            "## Safety & Accessibility Principles",
        ]
        for principle in self.safety_principles:
            lines.append(f"- {principle}")

        lines.append("\n## Weekly Schedule")
        for day in self.weekly_schedule:
            lines.append(f"\n### {day.day_label} — {day.focus}")
            lines.append("**Warm-Up:** " + "; ".join(day.warmup))
            lines.append("")
            for ex in day.exercises:
                cv_tag = " *(Supported in Exercise Analysis)*" if ex.cv_supported else ""
                lines.append(
                    f"- **{ex.name}**{cv_tag}: {ex.sets} sets × {ex.reps_or_duration} (Rest: {ex.rest_s}s)"
                )
                lines.append(f"  - *Form Cue:* {ex.coaching_cue}")
                lines.append(f"  - *Safety Modification:* {ex.safety_modification}")
            lines.append("\n**Cool-Down:** " + "; ".join(day.cooldown))

        lines.append("\n## Progression Guidance")
        for tip in self.progression_guidance:
            lines.append(f"- {tip}")

        return "\n".join(lines)


def _pick_squat_variant(profile: UserProfile) -> ExerciseItem:
    knee_friendly = "knee_friendly" in profile.accessibility_notes
    sets = 2 if profile.experience_level == "Beginner" else (3 if profile.experience_level == "Intermediate" else 4)

    if profile.goal == "Building Strength":
        reps = "8–10 controlled reps"
        rest = 90
    elif profile.goal == "Increasing Endurance":
        reps = "15–20 steady reps"
        rest = 45
    elif profile.goal == "Staying Active":
        reps = "10–12 comfortable reps"
        rest = 60
    else:
        reps = "10–15 reps"
        rest = 60

    if knee_friendly:
        return ExerciseItem(
            name="Box / Chair Squat (Joint-Friendly)",
            sets=sets,
            reps_or_duration=reps,
            rest_s=rest,
            coaching_cue="Sit hips back gently toward a sturdy chair or bench; keep chest proud and knees tracking over toes.",
            safety_modification="Use a higher seat height or hold a wall/railing for balance to reduce knee shear.",
            cv_supported=True,
        )

    if profile.equipment == "Full Gym" and profile.experience_level != "Beginner":
        name = "Goblet or Barbell Squat"
    elif profile.equipment == "Dumbbells & Bands" and profile.experience_level != "Beginner":
        name = "Goblet Squat"
    else:
        name = "Bodyweight Squat"

    return ExerciseItem(
        name=name,
        sets=sets,
        reps_or_duration=reps,
        rest_s=rest,
        coaching_cue="Keep heels planted, torso upright (trunk lean < 45°), and descend to a comfortable depth.",
        safety_modification="Limit depth to pain-free range or place a chair behind you as a depth target.",
        cv_supported=True,
    )


def _pick_curl_variant(profile: UserProfile) -> ExerciseItem:
    shoulder_friendly = "shoulder_friendly" in profile.accessibility_notes
    sets = 2 if profile.experience_level == "Beginner" else 3

    if profile.goal == "Building Strength":
        reps = "8–10 reps"
        rest = 75
    elif profile.goal == "Increasing Endurance":
        reps = "15–18 reps"
        rest = 45
    else:
        reps = "10–12 reps"
        rest = 60

    if profile.equipment == "Bodyweight Only":
        name = "Towel / Isometric Bicep Curl"
    elif profile.equipment == "Dumbbells & Bands":
        name = "Dumbbell or Band Bicep Curl"
    else:
        name = "Dumbbell / Cable Bicep Curl"

    cue = "Keep elbows pinned near your ribs and avoid swinging your shoulders during the curl."
    mod = (
        "Perform seated with back support to stabilize the shoulder joint."
        if shoulder_friendly
        else "Use a lighter load and slow 2-second lowering phase if you feel wrist or elbow strain."
    )
    return ExerciseItem(
        name=name,
        sets=sets,
        reps_or_duration=reps,
        rest_s=rest,
        coaching_cue=cue,
        safety_modification=mod,
        cv_supported=True,
    )


def _pick_push_variant(profile: UserProfile) -> ExerciseItem:
    shoulder_friendly = "shoulder_friendly" in profile.accessibility_notes
    sets = 2 if profile.experience_level == "Beginner" else 3
    reps = "8–12 reps" if profile.goal != "Increasing Endurance" else "12–16 reps"
    rest = 75 if profile.goal == "Building Strength" else 60

    if shoulder_friendly or profile.experience_level == "Beginner":
        return ExerciseItem(
            name="Wall or Incline Push-Up",
            sets=sets,
            reps_or_duration=reps,
            rest_s=rest,
            coaching_cue="Keep body in a straight line from head to heels and tuck elbows ~45° from your sides.",
            safety_modification="Elevate hands higher on a wall or counter to reduce shoulder and wrist pressure.",
        )

    name = "Dumbbell Floor / Bench Press" if profile.equipment != "Bodyweight Only" else "Standard Push-Up"
    return ExerciseItem(
        name=name,
        sets=sets,
        reps_or_duration=reps,
        rest_s=rest,
        coaching_cue="Lower under control until upper arms are parallel to the torso, then press smoothly.",
        safety_modification="Drop to knees or switch to an incline angle if lower back sags.",
    )


def _pick_posterior_chain_variant(profile: UserProfile) -> ExerciseItem:
    back_friendly = "back_friendly" in profile.accessibility_notes
    sets = 2 if profile.experience_level == "Beginner" else 3
    reps = "10–12 reps" if profile.goal != "Increasing Endurance" else "15 reps"
    rest = 60

    if back_friendly or profile.equipment == "Bodyweight Only":
        return ExerciseItem(
            name="Glute Bridge",
            sets=sets,
            reps_or_duration=reps,
            rest_s=rest,
            coaching_cue="Drive through your heels, squeeze glutes at the top, and avoid over-arching your lower back.",
            safety_modification="Keep range of motion moderate and place feet slightly wider for hip comfort.",
        )

    return ExerciseItem(
        name="Hip Hinge / Romanian Deadlift (Light–Moderate)",
        sets=sets,
        reps_or_duration=reps,
        rest_s=rest,
        coaching_cue="Push hips straight back with a neutral spine and soft knees.",
        safety_modification="Switch to Glute Bridges if you feel any lower-back tightness.",
    )


def _pick_conditioning_variant(profile: UserProfile) -> ExerciseItem:
    knee_friendly = "knee_friendly" in profile.accessibility_notes
    if profile.goal == "Increasing Endurance":
        duration = "12–18 min steady pace"
    elif profile.goal == "Staying Active":
        duration = "10–15 min brisk, conversational pace"
    else:
        duration = "8–12 min moderate pace"

    if knee_friendly:
        return ExerciseItem(
            name="Low-Impact Cycling / Rowing / Brisk Walking",
            sets=1,
            reps_or_duration=duration,
            rest_s=0,
            coaching_cue="Maintain upright posture and smooth breathing (able to speak in short sentences).",
            safety_modification="Keep resistance light-to-moderate to protect knee joints.",
        )

    return ExerciseItem(
        name="Brisk Walking / Step-Jacks / Low-Impact Intervals",
        sets=1,
        reps_or_duration=duration,
        rest_s=0,
        coaching_cue="Alternate 45s steady effort with 30s easy recovery pace.",
        safety_modification="Replace any jumping with step-touches for a zero-impact option.",
    )


def _pick_core_variant(profile: UserProfile) -> ExerciseItem:
    back_friendly = "back_friendly" in profile.accessibility_notes
    sets = 2 if profile.experience_level == "Beginner" else 3
    if back_friendly:
        return ExerciseItem(
            name="Dead Bug or Bird-Dog",
            sets=sets,
            reps_or_duration="8–10 slow reps per side",
            rest_s=45,
            coaching_cue="Brace your core gently and move opposite arm and leg without letting your lower back arch.",
            safety_modification="Keep knees bent or move only legs at first to keep spine supported.",
        )
    return ExerciseItem(
        name="Modified Plank (Knees or Forearms)",
        sets=sets,
        reps_or_duration="20–40 seconds",
        rest_s=45,
        coaching_cue="Keep ribs down, glutes engaged, and neck neutral looking at the floor.",
        safety_modification="Elevate hands on a bench or drop to knees to maintain clean spinal alignment.",
    )


def generate_recommendation_plan(profile: UserProfile) -> RecommendationPlan:
    """Generate a personalized, safety-first exercise recommendation plan."""
    weight_kg = max(25.0, float(profile.weight_kg))
    height_cm = max(100.0, float(profile.height_cm))
    days = max(2, min(6, int(profile.days_per_week)))
    minutes = max(15, min(90, int(profile.session_minutes)))

    safety_principles = [
        "Focus on consistency, joint comfort, and movement quality rather than body-weight or appearance targets.",
        "Stop any movement immediately if you feel sharp pain, dizziness, or joint pinching; mild muscle fatigue is normal.",
        "Use a controlled tempo (2 seconds lowering, 1 second lifting) and breathe steadily throughout each repetition.",
    ]

    if profile.accessibility_notes:
        readable_notes = [n.replace("_", " ").title() for n in profile.accessibility_notes]
        safety_principles.append(
            f"Joint-friendly adaptations applied for: {', '.join(readable_notes)}."
        )

    if weight_kg >= 100.0 and "knee_friendly" not in profile.accessibility_notes:
        safety_principles.append(
            "Low-impact warm-ups and controlled joint loading are emphasized to keep ankles, knees, and hips comfortable."
        )

    warmup = [
        "3–5 min gentle marching in place or easy walking",
        "Arm circles, shoulder rolls, and torso rotations (30s each)",
        "5 slow bodyweight half-squats or hip hinges to lubricate joints",
    ]
    cooldown = [
        "2–3 min slow walking + deep diaphragmatic breathing",
        "Gentle calf, hamstring, chest, and upper-back stretches (20–30s each, pain-free)",
    ]

    squat = _pick_squat_variant(profile)
    curl = _pick_curl_variant(profile)
    push = _pick_push_variant(profile)
    posterior = _pick_posterior_chain_variant(profile)
    cardio = _pick_conditioning_variant(profile)
    core = _pick_core_variant(profile)

    day_templates = [
        WorkoutDay(
            day_label="Day 1",
            focus="Lower Body & Core Stability",
            warmup=warmup,
            exercises=[squat, posterior, core] + ([cardio] if minutes >= 35 else []),
            cooldown=cooldown,
        ),
        WorkoutDay(
            day_label="Day 2",
            focus="Upper Body & Posture",
            warmup=warmup,
            exercises=[push, curl, core],
            cooldown=cooldown,
        ),
        WorkoutDay(
            day_label="Day 3",
            focus="Full-Body Conditioning & Mobility",
            warmup=warmup,
            exercises=[squat, curl, cardio],
            cooldown=cooldown,
        ),
        WorkoutDay(
            day_label="Day 4",
            focus="Strength & Joint Control",
            warmup=warmup,
            exercises=[posterior, push, core],
            cooldown=cooldown,
        ),
        WorkoutDay(
            day_label="Day 5",
            focus="Active Endurance & Balance",
            warmup=warmup,
            exercises=[squat, curl, cardio],
            cooldown=cooldown,
        ),
        WorkoutDay(
            day_label="Day 6",
            focus="Light Recovery Movement",
            warmup=warmup,
            exercises=[cardio, core],
            cooldown=cooldown,
        ),
    ]

    schedule = day_templates[:days]

    progression_guidance = [
        "Master clean form first: aim for 80%+ form consistency in the Exercise Analysis tab before increasing difficulty.",
        "Progress one variable at a time—add 1–2 reps per set or 5 minutes of conditioning only after a week feels comfortable.",
        "Schedule at least 1–2 rest or gentle walking days between harder sessions to support recovery.",
    ]

    return RecommendationPlan(
        profile_summary={
            "weight_kg": round(weight_kg, 1),
            "height_cm": round(height_cm, 1),
            "gender": profile.gender,
            "goal": profile.goal,
            "experience_level": profile.experience_level,
            "days_per_week": days,
            "session_minutes": minutes,
            "equipment": profile.equipment,
            "accessibility_notes": list(profile.accessibility_notes),
        },
        safety_principles=safety_principles,
        weekly_schedule=schedule,
        progression_guidance=progression_guidance,
    )
