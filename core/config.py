"""Configuration loader and manager for thresholds and hyperparameters."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
import copy

DEFAULT_CONFIG: dict[str, Any] = {
    "smoothing": {
        "method": "ema",
        "alpha": 0.4,
        "window": 7,
    },
    "visibility": {
        "min_visibility": 0.5,
        "max_hold_frames": 10,
    },
    "squat": {
        "standing_min": 160.0,
        "descent_start": 150.0,
        "bottom_max": 100.0,
        "ascent_hysteresis": 10.0,
        "trunk_lean_max": 45.0,
        "count_incomplete_extension": False,
    },
    "bicep_curl": {
        "extended_min": 150.0,
        "curling_start": 140.0,
        "flexed_max": 50.0,
        "lowering_start": 60.0,
        "shoulder_drift_max": 0.08,
    },
    "scoring": {
        "weights": {
            "consistency": 0.3,
            "rom": 0.4,
            "stability": 0.3,
        }
    },
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file or return defaults.

    Args:
        config_path: Path to YAML config file. If None, tries 'config/thresholds.yaml'.

    Returns:
        Deep-merged dictionary with configuration values.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        default_yaml = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"
        if default_yaml.exists():
            config_path = default_yaml

    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
                _deep_update(config, user_cfg)

    return config


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    """Recursively update base dictionary with update dictionary."""
    for key, val in update.items():
        if isinstance(val, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
