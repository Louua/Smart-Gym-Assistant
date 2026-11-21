"""Smart Gym Assistant - Streamlit Entry Point (Home Page)."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
while _root.parent != _root and not (_root / "core").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
from app.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Smart Gym Assistant — Exercise Analysis & Fitness Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

st.title("🏋️ Smart Gym Assistant")
st.subheader("Explainable Computer Vision Exercise Analysis & Health-First Fitness Planning")

st.markdown(
    """
    **Academic Project:** Computer Engineering — *Visual Tracking & Intelligent Applications (2026-2027)*  
    **Authors:** Loua Ben Afia & Abir Rouag  
    **Supervisor:** Dr. Rihab Laajili, Polytechnic Institute
    """
)

st.divider()

st.markdown("## Choose a Section")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 🔬 1. Exercise Analysis")
    st.markdown(
        """
        Upload an **exercise image** (`.png`, `.jpg`) or **workout video file** (`.mp4`, `.avi`, `.mov`),
        or use a live webcam / demo clip.

        - **Pose & Angle Detection:** Extracts 12 biomechanical landmarks via MediaPipe Pose.
        - **Form Explanations & Solutions:** Diagnoses squat depth, trunk lean, elbow flexion, and repetition quality.
        - **Analytics & Export:** Interactive Plotly trajectories, quality radar charts, and CSV/JSON export.
        """
    )
    if st.button("Open Exercise Analysis →", type="primary", use_container_width=True):
        st.switch_page("pages/1_Exercise_Analysis.py")

with col2:
    st.markdown("### 🧭 2. Personalized Fitness Recommendations")
    st.markdown(
        """
        Enter your **weight, height, gender, fitness goal, experience level, and accessibility preferences**
        to generate a structured weekly exercise plan.

        - **Goal-Adapted:** Tailored for *Improving Fitness*, *Building Strength*, *Increasing Endurance*, or *Staying Active*.
        - **Safety & Accessibility First:** Includes joint-friendly modifications, coaching cues, and warm-up/cool-down routines.
        - **Health-Focused:** Prioritizes sustainable movement and overall wellbeing—never weight or appearance targets.
        """
    )
    if st.button("Open Fitness Recommendations →", type="primary", use_container_width=True):
        st.switch_page("pages/2_Fitness_Recommendations.py")

st.divider()

st.info(
    "**Camera Setup Tip:** For the most accurate Exercise Analysis, position your camera at waist/chest height "
    "in a clear **side (90°) or 45° view** with your full body visible and even lighting."
)
