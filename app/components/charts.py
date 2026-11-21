"""Plotly visualization components for workout analytics."""

from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
from core.models import FrameResult, RepEvent


def create_angle_trajectory_chart(
    frames: List[FrameResult],
    reps: List[RepEvent],
    exercise_name: str,
    thresholds: Dict[str, Any] | None = None,
) -> go.Figure:
    """Create interactive Plotly angle-over-time chart with threshold lines and rep regions."""
    if not frames:
        fig = go.Figure()
        fig.update_layout(title="No frame data available")
        return fig

    # Build DataFrame from frames
    records = []
    for f in frames:
        rec = {"time": f.timestamp_s, "state": f.state}
        rec.update(f.angles)
        records.append(rec)

    df = pd.DataFrame(records)
    fig = go.Figure()

    is_squat = "squat" in exercise_name.lower()
    main_angle_key = "knee" if is_squat else "elbow"

    # Primary joint angle curve
    if main_angle_key in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[main_angle_key],
                mode="lines",
                name=f"{main_angle_key.capitalize()} Angle (°)",
                line=dict(color="#00E5FF", width=3),
            )
        )

    # Secondary angle curve (Hip for squats, Shoulder for curls)
    sec_angle_key = "hip" if is_squat else "shoulder"
    if sec_angle_key in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[sec_angle_key],
                mode="lines",
                name=f"{sec_angle_key.capitalize()} Angle (°)",
                line=dict(color="#FF9100", width=2, dash="dot"),
            )
        )

    # Trunk inclination for squats
    if is_squat and "trunk_inclination" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["trunk_inclination"],
                mode="lines",
                name="Trunk Lean (°)",
                line=dict(color="#E040FB", width=1.5, dash="dash"),
            )
        )

    # Add threshold reference lines
    if thresholds:
        if is_squat:
            bottom_thresh = thresholds.get("bottom_max", 100.0)
            stand_thresh = thresholds.get("standing_min", 160.0)
            fig.add_hline(
                y=bottom_thresh,
                line_dash="dash",
                line_color="#00E676",
                annotation_text=f"Target Depth (≤{bottom_thresh:.0f}°)",
                annotation_position="bottom right",
            )
            fig.add_hline(
                y=stand_thresh,
                line_dash="dash",
                line_color="#76FF03",
                annotation_text=f"Full Lockout (≥{stand_thresh:.0f}°)",
                annotation_position="top right",
            )
        else:
            flex_thresh = thresholds.get("flexed_max", 50.0)
            ext_thresh = thresholds.get("extended_min", 150.0)
            fig.add_hline(
                y=flex_thresh,
                line_dash="dash",
                line_color="#00E676",
                annotation_text=f"Peak Flexion (≤{flex_thresh:.0f}°)",
                annotation_position="bottom right",
            )
            fig.add_hline(
                y=ext_thresh,
                line_dash="dash",
                line_color="#76FF03",
                annotation_text=f"Full Extension (≥{ext_thresh:.0f}°)",
                annotation_position="top right",
            )

    # Highlight repetition windows
    for r in reps:
        fill_color = "rgba(0, 230, 118, 0.15)" if r.valid else "rgba(255, 23, 68, 0.2)"
        border_color = "#00E676" if r.valid else "#FF1744"
        label = f"Rep {r.rep_index}" if r.valid else f"Reject {r.rep_index}"

        fig.add_vrect(
            x0=r.start_s,
            x1=r.end_s,
            fillcolor=fill_color,
            line=dict(color=border_color, width=1, dash="dash"),
            annotation_text=label,
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"{exercise_name} Biomechanical Trajectory Over Time",
        xaxis_title="Workout Time (seconds)",
        yaxis_title="Joint Angle (degrees)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_radar_score_chart(summary: Dict[str, float]) -> go.Figure:
    """Create radar / spider chart of workout quality scores (0-100)."""
    categories = ["Form Consistency", "Range of Motion", "Stability", "Overall Form"]
    values = [
        summary.get("form_consistency", 0.0),
        summary.get("range_of_motion", 0.0),
        summary.get("stability", 0.0),
        summary.get("overall_form", 0.0),
    ]
    # Close the radar polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(0, 229, 255, 0.35)",
            line=dict(color="#00E5FF", width=2.5),
            name="Session Scores",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color="#AAAAAA")),
            angularaxis=dict(tickfont=dict(size=12, color="#FFFFFF")),
        ),
        showlegend=False,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=30, b=30),
        height=320,
    )
    return fig


def create_gauge_chart(score: float, title: str = "Overall Form Score") -> go.Figure:
    """Create an attractive gauge meter for a 0-100 score."""
    color = "#00E676" if score >= 80 else ("#FFD600" if score >= 60 else "#FF1744")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 18, "color": "#FFFFFF"}},
            number={"suffix": "%", "font": {"size": 36, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888888"},
                "bar": {"color": color},
                "bgcolor": "#222222",
                "borderwidth": 1,
                "bordercolor": "#444444",
                "steps": [
                    {"range": [0, 60], "color": "rgba(255, 23, 68, 0.2)"},
                    {"range": [60, 80], "color": "rgba(255, 214, 0, 0.2)"},
                    {"range": [80, 100], "color": "rgba(0, 230, 118, 0.2)"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=260,
    )
    return fig
