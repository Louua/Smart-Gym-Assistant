"""Module 1: Video Input Layer.

Abstracts video capture across recorded files and live webcams.
Guarantees clean resource cleanup, accurate FPS detection, and optional aspect-preserving downscaling.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generator, Tuple, Any
from pathlib import Path
import time
import cv2
import numpy as np


class VideoOpenError(Exception):
    """Raised when a video source cannot be opened."""
    pass


class VideoSource(ABC):
    """Abstract base class for video inputs."""

    @abstractmethod
    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_idx, timestamp_s, bgr_image) for each frame."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Video frames per second."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> Tuple[int, int]:
        """(width, height) of frames."""
        pass

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total frame count if known, or -1 for live streams."""
        pass

    def release(self) -> None:
        """Release underlying hardware/file handles."""
        pass

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class FileVideoSource(VideoSource):
    """Reads video from MP4, AVI, or MOV files."""

    def __init__(self, file_path: str | Path, max_width: int = 960) -> None:
        self.file_path = str(file_path)
        self.max_width = max_width
        self._cap = cv2.VideoCapture(self.file_path)
        if not self._cap.isOpened():
            raise VideoOpenError(f"Failed to open video file: {self.file_path}")

        file_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(file_fps) if file_fps and file_fps > 0 else 30.0

        raw_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.max_width and raw_w > self.max_width:
            scale = self.max_width / raw_w
            self._width = int(raw_w * scale)
            self._height = int(raw_h * scale)
        else:
            self._width = raw_w
            self._height = raw_h

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def dimensions(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            # Calculate accurate timestamp from index and FPS
            timestamp_s = frame_idx / self._fps

            if (frame.shape[1], frame.shape[0]) != (self._width, self._height):
                frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)

            yield (frame_idx, timestamp_s, frame)
            frame_idx += 1

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class WebcamVideoSource(VideoSource):
    """Captures live video from connected webcam."""

    def __init__(self, device_index: int = 0, max_width: int = 960) -> None:
        self.device_index = device_index
        self.max_width = max_width
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise VideoOpenError(f"Failed to access webcam at device index {device_index}")

        file_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(file_fps) if file_fps and file_fps > 0 else 30.0

        raw_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.max_width and raw_w > self.max_width:
            scale = self.max_width / raw_w
            self._width = int(raw_w * scale)
            self._height = int(raw_h * scale)
        else:
            self._width = raw_w
            self._height = raw_h

        self._start_time: float | None = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def dimensions(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def frame_count(self) -> int:
        return -1

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        self._start_time = time.perf_counter()
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            timestamp_s = time.perf_counter() - self._start_time

            if (frame.shape[1], frame.shape[0]) != (self._width, self._height):
                frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)

            yield (frame_idx, timestamp_s, frame)
            frame_idx += 1

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
