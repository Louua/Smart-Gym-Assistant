"""Workout Setup & Configuration Page."""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is in sys.path
_root = Path(__file__).resolve().parent
while _root.parent != _root and not (_root / "core").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import tempfile
import streamlit as st
from app.components.sidebar import render_sidebar
from core.config import load_config

st.set_page_config(page_title="Workout Setup — Smart Gym Assistant", page_icon="⚙️", layout="wide")
render_sidebar()

st.title("⚙️ Workout Configuration")
st.markdown("Select your exercise, video input source, and adjust biomechanical detection thresholds.")

cfg = load_config()

col_main, col_thresh = st.columns([3, 2], gap="large")

with col_main:
    st.subheader("1. Exercise Selection")
    exercise_choice = st.radio(
        "Choose Exercise Type:",
        ["Squats", "Bicep Curls"],
        horizontal=True,
        index=0 if st.session_state.get("active_exercise") != "Bicep Curls" else 1,
    )
    st.session_state["active_exercise"] = exercise_choice

    st.subheader("2. Camera Perspective & Body Side")
    side_choice = st.selectbox(
        "Body Side to Track:",
        ["auto", "left", "right"],
        help="'auto' automatically selects the side with higher detection confidence (recommended for side views).",
        index=0,
    )
    st.session_state["side_choice"] = side_choice

    st.subheader("3. Video Input Source")
    input_type = st.radio(
        "Select Input Method:",
        ["Upload Video File (MP4/AVI/MOV)", "Sample / Demo Video", "Live Webcam"],
        index=1,
    )

    video_path = None
    is_webcam = False

    if input_type == "Upload Video File (MP4/AVI/MOV)":
        uploaded_file = st.file_uploader(
            "Upload workout video:",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV.",
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.success(f"Video uploaded: {uploaded_file.name}")

    elif input_type == "Sample / Demo Video":
        sample_dir = Path("data/videos")
        sample_dir.mkdir(parents=True, exist_ok=True)
        available_videos = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi"))

        video_names = [v.name for v in available_videos]
        if not video_names:
            st.warning("No sample videos found in `data/videos/`. Generate one below:")
            if st.button("Generate Synthetic Demo Videos"):
                with st.spinner("Rendering synthetic demo videos..."):
                    from scripts.generate_demo_video import generate_squat_video, generate_curl_video
                    generate_squat_video("data/videos/demo_squat.mp4", num_reps=3, include_shallow=True)
                    generate_curl_video("data/videos/demo_curl.mp4", num_reps=3, include_incomplete=True)
                    st.success("Demo videos created in `data/videos/`!")
                    st.rerun()
        else:
            default_idx = 0
            if exercise_choice == "Squats" and "demo_squat.mp4" in video_names:
                default_idx = video_names.index("demo_squat.mp4")
            elif exercise_choice == "Bicep Curls" and "demo_curl.mp4" in video_names:
                default_idx = video_names.index("demo_curl.mp4")

            selected_video = st.selectbox("Choose demo video:", video_names, index=default_idx)
            video_path = str(sample_dir / selected_video)

    elif input_type == "Live Webcam":
        is_webcam = True
        cam_index = st.number_input("Webcam Device Index:", min_value=0, max_value=5, value=0)
        st.session_state["cam_index"] = cam_index
        st.info("Live webcam will stream in real-time on the Analysis page.")

with col_thresh:
    st.subheader("4. Biomechanical Thresholds")
    st.caption("Configurable thresholds loaded from `config/thresholds.yaml`.")

    custom_cfg = dict(cfg)

    with st.expander("Movement Thresholds", expanded=True):
        if exercise_choice == "Squats":
            sq = cfg.get("squat", {})
            st.markdown("**Squat Angles (°)**")
            standing_min = st.slider("Standing Lockout Min (°)", 140, 180, int(sq.get("standing_min", 160)))
            descent_start = st.slider("Descent Trigger Angle (°)", 130, 170, int(sq.get("descent_start", 150)))
            bottom_max = st.slider("Target Bottom Depth Max (°)", 70, 120, int(sq.get("bottom_max", 100)))
            ascent_hyst = st.slider("Ascent Hysteresis (°)", 5, 20, int(sq.get("ascent_hysteresis", 10)))
            trunk_lean_max = st.slider("Trunk Lean Max (°)", 25, 60, int(sq.get("trunk_lean_max", 45)))

            custom_cfg["squat"] = {
                "standing_min": float(standing_min),
                "descent_start": float(descent_start),
                "bottom_max": float(bottom_max),
                "ascent_hysteresis": float(ascent_hyst),
                "trunk_lean_max": float(trunk_lean_max),
                "count_incomplete_extension": False,
            }
        else:
            bc = cfg.get("bicep_curl", {})
            st.markdown("**Bicep Curl Angles (°)**")
            ext_min = st.slider("Extended Arm Min (°)", 130, 180, int(bc.get("extended_min", 150)))
            curl_start = st.slider("Curling Trigger Angle (°)", 120, 160, int(bc.get("curling_start", 140)))
            flex_max = st.slider("Peak Flexion Max (°)", 30, 80, int(bc.get("flexed_max", 50)))
            lower_start = st.slider("Lowering Trigger Angle (°)", 40, 90, int(bc.get("lowering_start", 60)))
            drift_max = st.slider("Shoulder Drift Max (% Torso)", 0.04, 0.20, float(bc.get("shoulder_drift_max", 0.08)), step=0.01)

            custom_cfg["bicep_curl"] = {
                "extended_min": float(ext_min),
                "curling_start": float(curl_start),
                "flexed_max": float(flex_max),
                "lowering_start": float(lower_start),
                "shoulder_drift_max": float(drift_max),
            }

    with st.expander("Filter & Visibility Hyperparameters", expanded=False):
        alpha = st.slider("EMA Smoothing Alpha", 0.1, 0.9, float(cfg.get("smoothing", {}).get("alpha", 0.4)), step=0.05)
        min_vis = st.slider("Min Landmark Visibility", 0.3, 0.9, float(cfg.get("visibility", {}).get("min_visibility", 0.5)), step=0.05)
        custom_cfg["smoothing"]["alpha"] = float(alpha)
        custom_cfg["visibility"]["min_visibility"] = float(min_vis)

st.divider()

# Validation & Launch
ready = (video_path is not None) or is_webcam
if not ready:
    st.warning("Please upload a video, select a demo video, or choose webcam to proceed.")
else:
    if st.button("🚀 Start Workout Analysis", type="primary", use_container_width=True):
        st.session_state["workout_configured"] = True
        st.session_state["is_webcam"] = is_webcam
        st.session_state["video_path"] = video_path
        st.session_state["runtime_config"] = custom_cfg
        st.session_state["session_result"] = None
        st.switch_page("pages/2_Analysis.py")
