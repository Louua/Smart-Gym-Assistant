"""Exercise registry and factory."""

from __future__ import annotations
from typing import Dict, Any, Type
from core.exercises.base import Exercise
from core.exercises.squat import SquatExercise
from core.exercises.bicep_curl import BicepCurlExercise

EXERCISE_REGISTRY: Dict[str, Type[Exercise]] = {
    "Squats": SquatExercise,
    "Bicep Curls": BicepCurlExercise,
}


def create_exercise(exercise_name: str, config: Dict[str, Any] | None = None) -> Exercise:
    """Instantiate an exercise tracker by name."""
    cls = EXERCISE_REGISTRY.get(exercise_name)
    if cls is None:
        raise ValueError(
            f"Unknown exercise '{exercise_name}'. Available: {list(EXERCISE_REGISTRY.keys())}"
        )
    return cls(config=config)
