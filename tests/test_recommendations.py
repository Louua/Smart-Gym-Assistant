"""Unit tests for the personalized fitness recommendation engine."""

from core.recommendations import (
    UserProfile,
    VALID_GOALS,
    generate_recommendation_plan,
)


def test_plan_matches_requested_days_and_goal():
    for goal in VALID_GOALS:
        profile = UserProfile(
            weight_kg=72.0,
            height_cm=175.0,
            gender="Prefer not to say",
            goal=goal,
            experience_level="Beginner",
            days_per_week=4,
            session_minutes=30,
            equipment="Bodyweight Only",
        )
        plan = generate_recommendation_plan(profile)
        assert len(plan.weekly_schedule) == 4
        assert plan.profile_summary["goal"] == goal
        assert len(plan.safety_principles) >= 3
        md = plan.to_markdown()
        assert "Personalized Exercise Recommendation Plan" in md


def test_accessibility_modifications_applied():
    profile = UserProfile(
        weight_kg=85.0,
        height_cm=170.0,
        gender="Female",
        goal="Staying Active",
        experience_level="Beginner",
        days_per_week=3,
        session_minutes=30,
        equipment="Dumbbells & Bands",
        accessibility_notes=["knee_friendly", "back_friendly", "shoulder_friendly"],
    )
    plan = generate_recommendation_plan(profile)
    all_names = [ex.name for day in plan.weekly_schedule for ex in day.exercises]
    assert any("Box / Chair Squat" in name for name in all_names)
    assert any("Glute Bridge" in name for name in all_names)
    assert any("Wall or Incline Push-Up" in name for name in all_names)
    assert any("Joint-friendly adaptations applied" in p for p in plan.safety_principles)
