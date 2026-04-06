"""
Data Collection Entities - Production-ready.

Refactored 08-03-2026:
- Removed unused fields
- Clean, minimal structure
- Production-ready
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class VideoState(Enum):
    """Video recording state."""
    PENDING = "pending"
    RECORDING = "recording"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    REJECTED = "rejected"


class VideoType(Enum):
    """Video type (face angle)."""
    FRONTAL = "frontal"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class VideoInfo:
    """
    Information about a recorded video.

    Attributes:
        video_type: Type of video (frontal/horizontal/vertical)
        path: Path to video file
        duration: Video duration in seconds
        file_size: File size in bytes
        state: Current state
        quality_score: Quality score from Gate 1 (0-1, None if not checked)
    """
    video_type: str
    path: Optional[str] = None
    duration: float = 0.0
    file_size: int = 0
    state: VideoState = VideoState.PENDING
    quality_score: Optional[float] = None


__all__ = ["VideoInfo", "VideoState", "VideoType"]
