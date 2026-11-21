"""Section 2: Personalized Fitness Recommendations Page."""

from __future__ import annotations
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
while _root.parent != _root and not (_root / "core").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
from app.components.sidebar import render_sidebar
from core.recommendations import (
    UserProfile,
    VALID_GOALS,
    VALID_LEVELS,
    VALID_EQUIPMENT,
    generate_recommendation_plan,
)

st.set_page_config(
    page_title="Personalized Fitness Recommendations — Smart Gym Assistant",
    page_icon="🧭",
    layout="wide",
)
render_sidebar()

st.title("🧭 2. Personalized Fitness Recommendations")
st.markdown(
    "Enter your personal details, preferences, and fitness goals to generate a **safe, accessible, "
    "health-first exercise plan** tailored to your experience level."
)

with st.form("recommendation_form"):
    st.subheader("Personal Information & Goals")
    col1, col2, col3 = st.columns(3)

    with col1:
        weight_kg = st.number_input("Current Weight (kg):", min_value=30.0, max_value=250.0, value=70.0, step=0.5)
        height_cm = st.number_input("Height (cm):", min_value=120.0, max_value=230.0, value=172.0, step=1.0)
    with col2:
        gender = st.selectbox("Gender:", ["Prefer not to say", "Female", "Male", "Non-binary"])
        goal = st.selectbox("Primary Fitness Goal:", list(VALID_GOALS))
    with col3:
        experience = st.selectbox("Experience Level:", list(VALID_LEVELS))
        equipment = st.selectbox("Available Equipment:", list(VALID_EQUIPMENT))

    st.subheader("Schedule & Accessibility Preferences")
    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    with col_s1:
        days_per_week = st.slider("Days per Week:", min_value=2, max_value=6, value=3)
    with col_s2:
        session_minutes = st.select_slider("Minutes per Session:", options=[20, 30, 45, 60], value=30)
    with col_s3:
        st.markdown("**Joint-Friendly / Accessibility Options:**")
        knee_friendly = st.checkbox("Knee-friendly / Low-impact adaptations")
        back_friendly = st.checkbox("Lower-back-friendly adaptations")
        shoulder_friendly = st.checkbox("Shoulder-friendly adaptations")

    submitted = st.form_submit_button("✨ Generate Personalized Plan", type="primary", use_container_width=True)

if submitted:
    notes = []
    if knee_friendly:
        notes.append("knee_friendly")
    if back_friendly:
        notes.append("back_friendly")
    if shoulder_friendly:
        notes.append("shoulder_friendly")

    profile = UserProfile(
        weight_kg=weight_kg,
        height_cm=height_cm,
        gender=gender,
        goal=goal,
        experience_level=experience,
        days_per_week=days_per_week,
        session_minutes=session_minutes,
        equipment=equipment,
        accessibility_notes=notes,
    )
    st.session_state["recommendation_plan"] = generate_recommendation_plan(profile)

plan = st.session_state.get("recommendation_plan")
if plan is not None:
    st.divider()
    st.subheader("📋 Your Personalized Exercise Plan")

    st.success(
        f"**Goal:** {plan.profile_summary['goal']} • "
        f"**Level:** {plan.profile_summary['experience_level']} • "
        f"**Schedule:** {plan.profile_summary['days_per_week']} days/week (~{plan.profile_summary['session_minutes']} min) • "
        f"**Equipment:** {plan.profile_summary['equipment']}"
    )

    with st.expander("🛡️ Safety & Accessibility Principles", expanded=True):
        for p in plan.safety_principles:
            st.markdown(f"- {p}")

    for day in plan.weekly_schedule:
        with st.expander(f"📅 {day.day_label}: {day.focus}", expanded=True):
            st.markdown("**Warm-Up:** " + " • ".join(day.warmup))
            st.markdown("---")
            for ex in day.exercises:
                badge = " `🔬 CV Form Check Available`" if ex.cv_supported else ""
                st.markdown(
                    f"**{ex.name}**{badge} — **{ex.sets} sets × {ex.reps_or_duration}** "
                    f"(Rest: `{ex.rest_s}s`)"
                )
                st.caption(f"**Form Cue:** {ex.coaching_cue}")
                st.caption(f"**Safety Modification:** {ex.safety_modification}")
            st.markdown("---")
            st.markdown("**Cool-Down:** " + " • ".join(day.cooldown))

    st.subheader("📈 Progression & Next Steps")
    for tip in plan.progression_guidance:
        st.markdown(f"- {tip}")

    c_dl1, c_dl2, c_nav = st.columns([1, 1, 1])
    with c_dl1:
        st.download_button(
            "⬇️ Download Plan (Markdown)",
            data=plan.to_markdown().encode("utf-8"),
            file_name="fitness_recommendation_plan.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with c_dl2:
        st.download_button(
            "⬇️ Download Plan (JSON)",
            data=plan.to_json(indent=2).encode("utf-8"),
            file_name="fitness_recommendation_plan.json",
            mime="application/json",
            use_container_width=True,
        )
    with c_nav:
        if st.button("🔬 Check Form in Exercise Analysis", use_container_width=True):
            st.switch_page("pages/1_Exercise_Analysis.py")
