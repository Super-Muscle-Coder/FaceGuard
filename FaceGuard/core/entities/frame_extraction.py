"""
Frame Extraction Entities - Production-ready.

Refactored 08-03-2026:
- Removed unused fields (face_landmarks - Haar doesn't support)
- Simplified structure
- Production-ready
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class FrameQuality:
    """
    Frame quality information with image data.

    Attributes:
        frame_num: Frame number
        timestamp: Timestamp in seconds
        blur_score: Laplacian variance (higher = sharper)
        brightness: Average brightness (0-255)
        contrast: Standard deviation (contrast measure)
        exposure_status: 0=underexposed, 1=good, 2=overexposed
        has_face: Face detected?
        face_bbox: Face bounding box [x1, y1, x2, y2]
        face_size: Face size (width, height)
        face_confidence: Detection confidence (0-1, always 1.0 for Haar)
        snr: Signal-to-Noise Ratio (combined, in dB)
        snr_frequency: Frequency-based SNR (in dB)
        frame_data: Frame image data (BGR)
    """
    frame_num: int
    timestamp: float
    blur_score: float
    brightness: float
    contrast: float
    exposure_status: int
    has_face: bool
    face_bbox: list[int]  # [x1, y1, x2, y2]
    face_size: Tuple[int, int]  # (width, height)
    face_confidence: float  # Always 1.0 for Haar Cascade
    snr: float
    snr_frequency: float
    frame_data: np.ndarray


@dataclass
class VideoInfo:
    """Video metadata for frame extraction."""
    path: str
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int

    def __str__(self) -> str:
        return (
            f"FPS: {self.fps:.1f}, Frames: {self.total_frames}, "
            f"Duration: {self.duration:.1f}s, Resolution: {self.width}x{self.height}"
        )


__all__ = ["FrameQuality", "VideoInfo"]
