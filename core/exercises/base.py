"""Base class and FSM abstractions for exercise analysis.

Ensures strict separation of mathematical trajectory assessment from presentation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from core.models import PoseFrame, FrameResult, RepEvent


class Exercise(ABC):
    """Abstract base class for all exercise trackers."""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self.reps: List[RepEvent] = []
        self.rejected_attempts: int = 0
        self.current_state: str = "INIT"
        self._rep_start_s: float = 0.0
        self._rep_min_angle: float = 999.0
        self._rep_max_angle: float = -999.0
        self._rep_issues: set[str] = set()

    @property
    def rep_count(self) -> int:
        """Count of valid completed repetitions."""
        return sum(1 for r in self.reps if r.valid)

    @abstractmethod
    def update(
        self,
        frame: PoseFrame,
        image_width: int,
        image_height: int,
        side_override: str = "auto",
    ) -> FrameResult:
        """Process a single frame and update the repetition FSM."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset state and counters for a new workout."""
        self.reps.clear()
        self.rejected_attempts = 0
        self.current_state = "INIT"
        self._rep_start_s = 0.0
        self._rep_min_angle = 999.0
        self._rep_max_angle = -999.0
        self._rep_issues.clear()
