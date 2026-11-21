"""Smart Gym Assistant - Streamlit Entry Point (Home Page)."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_root = Path(__file__).resolve().parent
while _root.parent != _root and not (_root / "core").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
from app.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Smart Gym Assistant — AI Workout Tracker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

st.title("🏋️ Smart Gym Assistant")
st.subheader("AI Workout Tracker Using Computer Vision & Biomechanical FSM")

st.markdown(
    """
    **Academic Project:** Computer Engineering — *Visual Tracking & Intelligent Applications (2026-2027)*  
    **Authors:** Loua Ben Afia & Abir Rouag  
    **Supervisor:** Dr. Rihab Laajili, Polytechnic Institute  
    """
)

st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🎯 Project Motivation & Purpose")
    st.markdown(
        """
        Solo fitness training frequently suffers from common pitfalls:
        - **Lost counts:** Tracking repetitions mid-set during high-effort exhaustion.
        - **Form degradation:** Lack of immediate feedback leads to injury or diminished efficiency.
        - **Inconsistency:** Difficulty measuring biomechanical progress over time without costly personal coaches.

        The **Smart Gym Assistant** addresses this by transforming standard camera video (from any smartphone, webcam, or video file)
        into **structured biomechanical intelligence** (joint angles, repetition trajectories, and deterministic form coaching).
        """
    )

    st.markdown("### 🔬 Architectural Philosophy: Transparent & Explainable")
    st.markdown(
        """
        Unlike opaque "black-box" fitness apps, this pipeline is designed to **demystify the computer vision and tracking layers**:
        - **5 Decoupled Modules:** Video I/O → Pose Estimation → Temporal Tracking → Exercise Analysis → Dashboard.
        - **Deterministic Finite State Machines (FSM):** Reps are counted **only** if the full movement sequence is executed in order. Incomplete movements that reverse direction prematurely are cleanly rejected and diagnosed.
        - **No ML Hallucinations:** Every feedback cue is computed from pure trigonometric formulas and configurable thresholds.
        """
    )

with col2:
    st.info(
        """
        #### 📐 Camera Setup Best Practices
        
        To achieve maximum tracking fidelity:
        1. **Camera Angle:** Position camera at approximately waist/chest height in **profile view** (90° side view) or 45° oblique view.
        2. **Full Body Visibility:** Ensure head, torso, knees, and feet are fully framed.
        3. **Lighting:** Frontal/ambient illumination works best. Avoid bright backlights or deep shadows.
        4. **Clothing:** Form-fitting athletic wear provides cleanest joint landmark localization.
        """
    )

    st.success(
        """
        #### 🚀 Quickstart Journey
        
        1. Navigate to **[1_Workout](pages/1_Workout.py)** to select your exercise (Squats or Bicep Curls) and input source.
        2. Go to **[2_Analysis](pages/2_Analysis.py)** to observe the real-time skeletal HUD, live angles, and rep counter.
        3. Review your **[3_Analytics](pages/3_Analytics.py)** for interactive angle trajectories, quality radar charts, and CSV/JSON export.
        """
    )

st.divider()

st.markdown("### 🧩 Pipeline Overview")
st.markdown(
    """
    | Stage | Module | Key Technology | Primary Output |
    |---|---|---|---|
    | **1. Video Input** | `core/video_input.py` | OpenCV (`cv2.VideoCapture`) | Raw RGB/BGR frame stream, accurate FPS detection |
    | **2. Pose Estimation** | `core/pose_estimator.py` | MediaPipe Pose (`0.10.14`) | 12 core landmark coordinates (shoulders, elbows, wrists, hips, knees, ankles) |
    | **3. Temporal Tracking** | `core/tracker.py` | EMA Filter & Occlusion Hold | Smoothed trajectories, jitter rejection, missing keypoint carry-forward |
    | **4. Exercise Analysis** | `core/exercises/` | Biomechanical FSM + Trigonometry | Joint angles (knee, hip, elbow, trunk lean), FSM state progression, rep validation |
    | **5. Dashboard** | `app/` | Streamlit + Plotly | Real-time video player, HUD overlay, post-session analytics & data export |
    """
)
