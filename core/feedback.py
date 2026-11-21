"""Rule-based feedback translation engine.

Translates discrete biomechanical issues into user-facing coaching messages.
Strictly deterministic and explainable: no black box ML.
"""

from __future__ import annotations

ISSUE_MESSAGES: dict[str, str] = {
    "insufficient_depth": "Go lower — squat depth not reached",
    "incomplete_extension": "Stand up fully / extend your arm fully",
    "shoulder_compensation": "Keep your shoulder still — avoid swinging",
    "excessive_trunk_lean": "Keep your chest up — reduce forward lean",
    "incomplete_flexion": "Curl higher — complete peak contraction",
    "low_visibility": "Body not fully visible — adjust the camera",
}


def get_feedback_message(issue_code: str) -> str:
    """Return coaching message for a given issue code."""
    return ISSUE_MESSAGES.get(issue_code, f"Form issue: {issue_code}")


def format_feedback_list(issue_codes: list[str]) -> list[str]:
    """Translate a list of issue codes into user-friendly messages."""
    return [get_feedback_message(code) for code in issue_codes]
