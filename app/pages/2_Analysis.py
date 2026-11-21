"""Real-time Exercise Analysis & HUD Overlay Page."""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is in sys.path
_root = Path(__file__).resolve().parent
while _root.parent != _root and not (_root / "core").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
import cv2
import streamlit as st
from app.components.sidebar import render_sidebar
from core.video_input import FileVideoSource, WebcamVideoSource, VideoOpenError
from core.pose_estimator import PoseEstimator
from core.exercises import create_exercise
from core.pipeline import WorkoutPipeline
from core.config import load_config

st.set_page_config(page_title="Live Analysis — Smart Gym Assistant", page_icon="📹", layout="wide")
render_sidebar()

st.title("📹 Real-Time Biomechanical Analysis")

if not st.session_state.get("workout_configured", False):
    st.warning("⚠️ No workout session is currently configured.")
    if st.button("Go to Workout Setup"):
        st.switch_page("pages/1_Workout.py")
    st.stop()

# Retrieve parameters
exercise_name = st.session_state.get("active_exercise", "Squats")
is_webcam = st.session_state.get("is_webcam", False)
video_path = st.session_state.get("video_path")
side_override = st.session_state.get("side_choice", "auto")
runtime_cfg = st.session_state.get("runtime_config") or load_config()

st.markdown(f"**Exercise:** `{exercise_name}` | **Body Side:** `{side_override}` | **Source:** `{'Live Webcam' if is_webcam else video_path}`")

# Header controls
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
with btn_col1:
    start_clicked = st.button("▶️ Start Tracking", type="primary", use_container_width=True)
with btn_col2:
    back_clicked = st.button("⚙️ Edit Setup", use_container_width=True)

if back_clicked:
    st.switch_page("pages/1_Workout.py")

# Layout: video stream on the left, live telemetry on the right
col_video, col_telemetry = st.columns([7, 3], gap="medium")

with col_telemetry:
    st.markdown("### 📊 Live Telemetry")
    rep_kpi = st.empty()
    state_kpi = st.empty()
    st.divider()
    angles_container = st.empty()
    st.divider()
    feedback_container = st.empty()

with col_video:
    video_placeholder = st.empty()
    stop_btn_placeholder = st.empty()

# Persistent state placeholder for finished session
if st.session_state.get("session_result") is not None and not start_clicked:
    res = st.session_state["session_result"]
    valid_reps = len([r for r in res.reps if r.valid])
    st.success(f"🎉 Session completed! Recorded **{valid_reps} valid reps** ({res.rejected_attempts} rejected attempts).")
    if st.button("📈 Open Analytics & Performance Report", type="primary"):
        st.switch_page("pages/3_Analytics.py")

if start_clicked:
    # Initialize video source
    try:
        if is_webcam:
            cam_idx = st.session_state.get("cam_index", 0)
            source = WebcamVideoSource(device_index=cam_idx)
        else:
            source = FileVideoSource(video_path)
    except VideoOpenError as e:
        st.error(f"Failed to open video source: {e}")
        st.stop()

    exercise = create_exercise(exercise_name, config=runtime_cfg)
    pipeline = WorkoutPipeline(
        exercise=exercise,
        source=source,
        config=runtime_cfg,
        side_override=side_override,
    )

    stop_stream = False
    stop_button = stop_btn_placeholder.button("⏹️ Stop & Save Session", use_container_width=True)

    progress_bar = st.progress(0.0)
    total_frames = source.frame_count if source.frame_count > 0 else 1

    try:
        for annotated_bgr, frame_res in pipeline.run():
            # Convert BGR to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Update Rep KPIs
            rep_kpi.markdown(
                f"""
                <div style="background-color: #1E1E1E; padding: 15px; border-radius: 8px; text-align: center;">
                    <span style="font-size: 14px; color: #888888; text-transform: uppercase;">Valid Repetitions</span>
                    <h1 style="color: #00E676; margin: 0; font-size: 44px;">{frame_res.rep_count}</h1>
                    <span style="font-size: 13px; color: #FF5252;">Rejected: {exercise.rejected_attempts}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Update State Badge
            state_colors = {
                "STANDING": "#4CAF50",
                "DESCENDING": "#FF9800",
                "BOTTOM": "#E91E63",
                "ASCENDING": "#2196F3",
                "EXTENDED": "#4CAF50",
                "CURLING": "#FF9800",
                "FLEXED": "#E91E63",
                "LOWERING": "#2196F3",
            }
            curr_color = state_colors.get(frame_res.state, "#9C27B0")
            state_kpi.markdown(
                f"""
                <div style="background-color: {curr_color}22; border: 2px solid {curr_color}; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px;">
                    <span style="font-size: 12px; color: #CCCCCC; text-transform: uppercase;">Current State</span>
                    <h3 style="color: {curr_color}; margin: 2px 0 0 0;">{frame_res.state}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Update Angle Readouts
            angle_md = "#### 📐 Live Joint Angles\n"
            for k, v in frame_res.angles.items():
                if not (v != v):  # Not NaN
                    angle_md += f"- **{k.capitalize()}:** `{v:.1f}°`\n"
            angles_container.markdown(angle_md)

            # Update Feedback Messages
            if frame_res.active_feedback:
                fb_md = "#### ⚠️ Real-Time Form Alerts\n"
                for msg in frame_res.active_feedback:
                    fb_md += f"<div style='background-color: rgba(255, 82, 82, 0.2); border-left: 4px solid #FF5252; padding: 6px 10px; margin-bottom: 6px; border-radius: 4px; font-size: 14px;'>{msg}</div>"
                feedback_container.markdown(fb_md, unsafe_allow_html=True)
            else:
                feedback_container.markdown(
                    "<div style='color: #4CAF50; font-size: 14px;'>✅ Clean execution — no active form issues</div>",
                    unsafe_allow_html=True,
                )

            # Update progress if file source
            if source.frame_count > 0:
                prog = min(1.0, max(0.0, frame_res.frame_idx / source.frame_count))
                progress_bar.progress(prog)

            # Check stop button (via session flag or brief sleep)
            time.sleep(0.005)

    except Exception as e:
        st.error(f"Error during video processing: {e}")
    finally:
        source.release()

    # Session completed: compute final summary
    session_result = pipeline.compute_session_result()
    st.session_state["session_result"] = session_result
    progress_bar.progress(1.0)

    st.balloons()
    st.success("🎉 Workout session completed successfully!")
    if st.button("View Detailed Performance Analytics 📈", type="primary"):
        st.switch_page("pages/3_Analytics.py")
