"""Global sidebar navigation and session status display."""

from __future__ import annotations
import streamlit as st


def render_sidebar() -> None:
    """Render unified navigation and status sidebar."""
    with st.sidebar:
        st.markdown("### 🏋️ **Smart Gym Assistant**")
        st.caption("AI Workout Tracker Using Computer Vision")
        st.divider()

        # Session state overview
        session_result = st.session_state.get("session_result")
        active_exercise = st.session_state.get("active_exercise", "Squats")

        st.markdown("#### 📊 **Status**")
        st.write(f"**Selected:** `{active_exercise}`")

        if session_result is not None:
            valid_count = len([r for r in session_result.reps if r.valid])
            st.success(f"Session complete: **{valid_count} reps**")
            st.caption(f"Rejected: {session_result.rejected_attempts}")
            st.caption(f"Score: {session_result.summary.get('overall_form', 0.0):.1f}%")
        else:
            st.info("No completed session in memory")

        st.divider()
        st.markdown("#### 🧭 **Quick Pages**")
        st.page_link("Home.py", label="Home & Setup Guide", icon="🏠")
        st.page_link("pages/1_Workout.py", label="Workout Setup", icon="⚙️")
        st.page_link("pages/2_Analysis.py", label="Live Tracker", icon="📹")
        st.page_link("pages/3_Analytics.py", label="Performance Analytics", icon="📈")

        st.divider()
        st.caption("Polytechnic Institute • 2026-2027")
        st.caption("Loua Ben Afia & Abir Rouag")
