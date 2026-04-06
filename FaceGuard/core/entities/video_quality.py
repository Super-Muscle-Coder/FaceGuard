"""Video Quality Entities.

===================  LỊCH SỬ REFACTOR  ====================
Refactored 30-01-2026:
- Thêm warnings field: Support WARNING level feedback
- Mục đích: User feedback khi quality borderline (giữa WARNING và CRITICAL)
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

VideoStatus = Literal["copied", "processed", "failed"]

@dataclass
class VideoQualityReport:
    """Báo cáo chất lượng video với strict thresholds."""
    path: str
    duration: float
    fps: float
    resolution: Tuple[int, int]
    total_frames: int
    avg_exposure_clip: float
    avg_blur_score: float
    snr: float
    snr_frequency: float
    salt_pepper_ratio: float
    has_critical_exposure: bool
    has_critical_blur: bool
    has_critical_noise: bool
    is_valid: bool
    validation_issues: List[str]
    fixable_issues: List[str]
    warnings: List[str]  # ← NEW! Warning-level issues (borderline quality)


__all__ = ["VideoQualityReport", "VideoStatus"]
