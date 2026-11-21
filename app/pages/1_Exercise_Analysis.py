"""Section 1: Exercise Analysis — Image & Video Biomechanical Analysis."""

from __future__ import annotations
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
while _root.parent != _root and not (_root / "core").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.charts import (
    create_angle_trajectory_chart,
    create_radar_score_chart,
    create_gauge_chart,
)
from core.config import load_config
from core.exercises import create_exercise
from core.image_analysis import analyze_exercise_image
from core.pipeline import WorkoutPipeline
from core.video_input import FileVideoSource, WebcamVideoSource, VideoOpenError

st.set_page_config(
    page_title="Exercise Analysis — Smart Gym Assistant",
    page_icon="🔬",
    layout="wide",
)
render_sidebar()

st.title("🔬 1. Exercise Analysis")
st.markdown(
    "Upload an **exercise image** or **workout video file** (or use a sample video / webcam) "
    "to analyze movement biomechanics, detect form issues, and receive actionable coaching solutions."
)

cfg = load_config()

# --- Input & Configuration ---
col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 2, 2])
with col_cfg1:
    input_mode = st.radio(
        "Input Type:",
        ["Upload Image (PNG/JPG)", "Upload Video File (MP4/AVI/MOV)", "Sample Video", "Live Webcam"],
        horizontal=False,
    )
with col_cfg2:
    if input_mode == "Upload Image (PNG/JPG)":
        exercise_choice = st.selectbox(
            "Exercise Type:",
            ["Auto-Detect", "Squats", "Bicep Curls"],
        )
    else:
        exercise_choice = st.radio(
            "Exercise Type:",
            ["Squats", "Bicep Curls"],
            horizontal=True,
            index=0 if st.session_state.get("active_exercise") != "Bicep Curls" else 1,
        )
        st.session_state["active_exercise"] = exercise_choice
with col_cfg3:
    side_override = st.selectbox(
        "Body Side:",
        ["auto", "left", "right"],
        help="'auto' selects the side with stronger landmark visibility.",
    )

st.divider()

# =====================================================================
# MODE A: IMAGE ANALYSIS
# =====================================================================
if input_mode == "Upload Image (PNG/JPG)":
    uploaded_img = st.file_uploader(
        "Upload an exercise image:",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a clear full-body or upper-body photo of your exercise posture.",
    )

    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if bgr_img is None:
            st.error("Could not decode the uploaded image. Please try another PNG or JPG file.")
        else:
            with st.spinner("Analyzing posture and joint geometry..."):
                result = analyze_exercise_image(
                    bgr_img,
                    exercise_hint=exercise_choice,
                    side_override=side_override,
                )

            col_orig, col_ann = st.columns(2, gap="medium")
            with col_orig:
                st.markdown("#### Original Image")
                st.image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_ann:
                st.markdown("#### Skeletal & Angle Overlay")
                st.image(cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.divider()
            st.subheader("Analysis & Coaching Feedback")

            if result.detected:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Detected Exercise", result.detected_exercise)
                m2.metric("Knee Angle", f"{result.angles.get('knee_angle', 0.0)}°")
                m3.metric("Elbow Angle", f"{result.angles.get('elbow_angle', 0.0)}°")
                m4.metric("Trunk Lean", f"{result.angles.get('trunk_lean', 0.0)}°")

                st.info(f"**Posture Summary ({result.side_used} side):** {result.posture_summary}")
            else:
                st.warning(result.posture_summary)

            col_exp, col_sol = st.columns(2, gap="large")
            with col_exp:
                st.markdown("#### 📋 What We Observed")
                for exp in result.explanations:
                    st.markdown(f"- {exp}")
            with col_sol:
                st.markdown("#### 💡 Recommended Solutions & Cues")
                for sol in result.solutions:
                    st.markdown(f"- {sol}")

# =====================================================================
# MODE B: VIDEO / SAMPLE / WEBCAM ANALYSIS
# =====================================================================
else:
    video_path = None
    is_webcam = False
    cam_index = 0

    if input_mode == "Upload Video File (MP4/AVI/MOV)":
        uploaded_vid = st.file_uploader(
            "Upload a workout video:",
            type=["mp4", "avi", "mov", "mkv"],
        )
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            video_path = tfile.name
            st.success(f"Loaded video: {uploaded_vid.name}")

    elif input_mode == "Sample Video":
        sample_dir = Path("data/videos")
        sample_dir.mkdir(parents=True, exist_ok=True)
        available = sorted(list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi")))
        if not available:
            st.warning("No sample videos found in `data/videos/`.")
            if st.button("Generate Synthetic Demo Videos"):
                with st.spinner("Generating demo videos..."):
                    from scripts.generate_demo_video import generate_squat_video, generate_curl_video
                    generate_squat_video("data/videos/demo_squat.mp4", num_reps=3, include_shallow=True)
                    generate_curl_video("data/videos/demo_curl.mp4", num_reps=3, include_incomplete=True)
                    st.rerun()
        else:
            names = [v.name for v in available]
            default_idx = 0
            if exercise_choice == "Squats" and "demo_squat.mp4" in names:
                default_idx = names.index("demo_squat.mp4")
            elif exercise_choice == "Bicep Curls" and "demo_curl.mp4" in names:
                default_idx = names.index("demo_curl.mp4")
            chosen = st.selectbox("Select Sample Video:", names, index=default_idx)
            video_path = str(sample_dir / chosen)

    else:
        is_webcam = True
        cam_index = int(st.number_input("Webcam Device Index:", min_value=0, max_value=5, value=0))

    run_btn = st.button(
        "▶️ Start Exercise Analysis",
        type="primary",
        disabled=(not is_webcam and video_path is None),
    )

    if run_btn:
        try:
            source = WebcamVideoSource(device_index=cam_index) if is_webcam else FileVideoSource(video_path)
        except VideoOpenError as exc:
            st.error(f"Could not open video source: {exc}")
            st.stop()

        exercise = create_exercise(exercise_choice, config=cfg)
        pipeline = WorkoutPipeline(
            exercise=exercise,
            source=source,
            config=cfg,
            side_override=side_override,
        )

        col_vid, col_tel = st.columns([7, 3], gap="medium")
        with col_vid:
            video_placeholder = st.empty()
            progress_bar = st.progress(0.0)
        with col_tel:
            st.markdown("### 📊 Live Telemetry")
            rep_kpi = st.empty()
            state_kpi = st.empty()
            feedback_box = st.empty()

        total_frames = source.frame_count if source.frame_count > 0 else 1

        try:
            for annotated_bgr, frame_res in pipeline.run():
                rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb, channels="RGB", use_container_width=True)

                rep_kpi.metric(
                    "Valid Repetitions",
                    frame_res.rep_count,
                    delta=f"-{exercise.rejected_attempts} rejected" if exercise.rejected_attempts else None,
                    delta_color="inverse",
                )
                state_kpi.info(f"**FSM State:** `{frame_res.state}`")
                if frame_res.feedback:
                    feedback_box.warning("\n".join(f"• {msg}" for msg in frame_res.feedback))
                else:
                    feedback_box.success("Form looks clean!")

                if not is_webcam and total_frames > 1:
                    progress_bar.progress(min(1.0, (frame_res.frame_idx + 1) / total_frames))
        finally:
            source.release()
            session_result = pipeline.compute_session_result()
            st.session_state["session_result"] = session_result
            st.session_state["runtime_config"] = cfg

    # --- Post-Session Analytics & Explanations ---
    session_result = st.session_state.get("session_result")
    if session_result is not None:
        st.divider()
        st.subheader("📈 Session Analytics, Explanations & Export")

        reps = session_result.reps
        valid_reps = [r for r in reps if r.valid]
        summary = session_result.summary
        overall_score = summary.get("overall_form", 0.0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Valid Repetitions", len(valid_reps))
        k2.metric("Rejected Attempts", session_result.rejected_attempts)
        k3.metric("Overall Form Score", f"{overall_score:.1f}%")
        k4.metric("Throughput", f"{summary.get('effective_fps', 0.0):.1f} FPS")

        c_gauge, c_radar = st.columns(2)
        with c_gauge:
            st.plotly_chart(create_gauge_chart(overall_score, "Overall Form Quality"), use_container_width=True)
        with c_radar:
            st.plotly_chart(create_radar_score_chart(summary), use_container_width=True)

        thresholds = cfg.get("squat" if "squat" in session_result.exercise.lower() else "bicep_curl", {})
        fig_traj = create_angle_trajectory_chart(
            frames=session_result.frames,
            reps=reps,
            exercise_name=session_result.exercise,
            thresholds=thresholds,
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        if reps:
            rep_rows = [
                {
                    "Rep": r.rep_index,
                    "Valid": r.valid,
                    "Duration (s)": round(r.end_s - r.start_s, 2),
                    "Min Angle (°)": round(r.min_angle, 1),
                    "Max Angle (°)": round(r.max_angle, 1),
                    "ROM (°)": round(r.max_angle - r.min_angle, 1),
                    "Issues / Coaching Cues": "; ".join(r.issues) if r.issues else "None",
                }
                for r in reps
            ]
            df_reps = pd.DataFrame(rep_rows)
            st.dataframe(df_reps, use_container_width=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Download Reps CSV",
                    data=df_reps.to_csv(index=False).encode("utf-8"),
                    file_name="exercise_reps.csv",
                    mime="text/csv",
                )
            with dl2:
                st.download_button(
                    "⬇️ Download Session JSON",
                    data=session_result.to_json(indent=2).encode("utf-8"),
                    file_name="exercise_session.json",
                    mime="application/json",
                )
