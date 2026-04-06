from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class DetectedFace:
    """Detected face container for realtime runtime/packager."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    embedding: Optional[np.ndarray] = None
    identity: Optional[str] = None
    recognition_confidence: Optional[float] = None
    smooth_bbox: Optional[Tuple[int, int, int, int]] = None

    @property
    def is_known(self) -> bool:
        return self.identity not in (None, "Unknown")

    @property
    def display_bbox(self) -> Tuple[int, int, int, int]:
        return self.smooth_bbox if self.smooth_bbox is not None else self.bbox


@dataclass
class FrameStatistics:
    """Runtime frame statistics for realtime package UI/service."""
    frame_count: int = 0
    fps: float = 0.0
    faces_detected: int = 0
    known_faces: int = 0
    unknown_faces: int = 0
    avg_confidence: float = 0.0
    processing_time: float = 0.0


__all__ = ["DetectedFace", "FrameStatistics"]
