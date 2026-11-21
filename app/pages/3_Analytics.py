"""Workout Analytics, Visualizations, and Data Export Page."""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is in sys.path
_root = Path(__file__).resolve().parent
while _root.parent != _root and not (_root / "core").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import pandas as pd
import streamlit as st
from app.components.sidebar import render_sidebar
from app.components.charts import (
    create_angle_trajectory_chart,
    create_radar_score_chart,
    create_gauge_chart,
)

st.set_page_config(page_title="Workout Analytics — Smart Gym Assistant", page_icon="📈", layout="wide")
render_sidebar()

st.title("📈 Performance Analytics & Diagnostics")
st.markdown("Comprehensive biomechanical review, repetition breakdown, and structured data export.")

session_result = st.session_state.get("session_result")

if session_result is None:
    st.info("ℹ️ No completed workout session found in memory.")
    st.markdown("To view performance metrics, run a workout session on the **[Analysis Page](pages/2_Analysis.py)** or select a demo video in **[Workout Setup](pages/1_Workout.py)**.")
    if st.button("Start New Workout"):
        st.switch_page("pages/1_Workout.py")
    st.stop()

reps = session_result.reps
valid_reps = [r for r in reps if r.valid]
summary = session_result.summary
overall_score = summary.get("overall_form", 0.0)

# Section 1: KPI Cards
st.subheader("1. Workout Summary")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Valid Repetitions", len(valid_reps), delta=f"{len(reps)} attempts")
kpi2.metric("Rejected Attempts", session_result.rejected_attempts, delta_color="inverse")
kpi3.metric("Overall Form Score", f"{overall_score:.1f}%")
kpi4.metric("Avg Throughput", f"{summary.get('effective_fps', 0.0):.1f} FPS")
kpi5.metric("Mean Latency", f"{summary.get('mean_latency_ms', 0.0):.1f} ms")

st.divider()

# Section 2: Quality Radar & Gauge
st.subheader("2. Biomechanical Quality Scores")
col_gauge, col_radar, col_breakdown = st.columns([1, 1, 1])

with col_gauge:
    fig_gauge = create_gauge_chart(overall_score, "Overall Form Quality")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_radar:
    fig_radar = create_radar_score_chart(summary)
    st.plotly_chart(fig_radar, use_container_width=True)

with col_breakdown:
    st.markdown("#### Quality Metrics Breakdown")
    st.markdown(
        f"""
        - **Form Consistency ({summary.get('form_consistency', 0.0):.1f}%):**  
          Measures repeatability of peak depth and lockout angles across repetitions.
        - **Range of Motion ({summary.get('range_of_motion', 0.0):.1f}%):**  
          Compares achieved angular excursion against biomechanical target ROM.
        - **Stability & Control ({summary.get('stability', 0.0):.1f}%):**  
          Penalizes trunk inclination drift or excessive shoulder swinging.
        """
    )

st.divider()

# Section 3: Interactive Plotly Trajectory
st.subheader("3. Joint Angle Trajectories Over Time")
thresholds = st.session_state.get("runtime_config", {}).get(
    "squat" if "squat" in session_result.exercise.lower() else "bicep_curl", {}
)
fig_traj = create_angle_trajectory_chart(
    frames=session_result.frames,
    reps=reps,
    exercise_name=session_result.exercise,
    thresholds=thresholds,
)
st.plotly_chart(fig_traj, use_container_width=True)

st.divider()

# Section 4: Per-Repetition Audit Table
st.subheader("4. Detailed Repetition Audit Table")
if not reps:
    st.info("No repetition attempts recorded.")
else:
    rep_records = []
    for r in reps:
        duration = r.end_s - r.start_s
        rom = r.max_angle - r.min_angle
        issues_str = ", ".join(r.issues) if r.issues else "None (Clean)"
        rep_records.append(
            {
                "Rep #": r.rep_index,
                "Status": "✅ Valid" if r.valid else "❌ Rejected",
                "Start Time (s)": round(r.start_s, 2),
                "End Time (s)": round(r.end_s, 2),
                "Duration (s)": round(duration, 2),
                "Min Angle (°)": round(r.min_angle, 1),
                "Max Angle (°)": round(r.max_angle, 1),
                "Range of Motion (°)": round(rom, 1),
                "Issues Detected": issues_str,
            }
        )
    df_reps = pd.DataFrame(rep_records)
    st.dataframe(df_reps, use_container_width=True)

st.divider()

# Section 5: Data Export
st.subheader("5. Export Session Data")
col_exp1, col_exp2, _ = st.columns([1, 1, 2])

# CSV Rep Table
csv_data = df_reps.to_csv(index=False).encode("utf-8") if reps else b""
col_exp1.download_button(
    label="📥 Download Reps as CSV",
    data=csv_data,
    file_name=f"workout_reps_{session_result.exercise.lower().replace(' ', '_')}.csv",
    mime="text/csv",
    disabled=len(reps) == 0,
    use_container_width=True,
)

# Full JSON Session
json_data = session_result.to_json(indent=2).encode("utf-8")
col_exp2.download_button(
    label="📥 Download Full Session JSON",
    data=json_data,
    file_name=f"session_{session_result.exercise.lower().replace(' ', '_')}.json",
    mime="application/json",
    use_container_width=True,
)
