"""Tests for feedback translation rules."""

from core.feedback import get_feedback_message, format_feedback_list, ISSUE_MESSAGES


def test_all_issue_codes_have_clear_messages():
    expected_codes = [
        "insufficient_depth",
        "incomplete_extension",
        "shoulder_compensation",
        "excessive_trunk_lean",
        "incomplete_flexion",
        "low_visibility",
    ]
    for code in expected_codes:
        assert code in ISSUE_MESSAGES
        msg = get_feedback_message(code)
        assert len(msg) > 5
        assert code not in msg  # Should be friendly English text, not raw code


def test_format_feedback_list():
    codes = ["insufficient_depth", "excessive_trunk_lean"]
    msgs = format_feedback_list(codes)
    assert len(msgs) == 2
    assert "depth" in msgs[0].lower()
    assert "chest" in msgs[1].lower()
