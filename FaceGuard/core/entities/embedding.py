from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EmbeddingResult:
    """
    Kết quả embedding của một ảnh, kèm metadata chất lượng.

    Refactored 31-01-2026:
    - Add face_confidence_gate2: MediaPipe confidence từ Gate 2
    - Add snr_gate2: Frame SNR từ Gate 2
    - Add landmarks_quality: Landmarks validation score
    """
    person_name: str
    image_path: str
    image_type: str  # 'frontal', 'horizontal', 'vertical'
    embedding: Optional[np.ndarray]
    detection_score: float
    bbox: List[int]
    landmarks: Optional[np.ndarray]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    quality_score: float = 0.0
    embedding_norm: float = 0.0
    frame_metadata: Dict = field(default_factory=dict)

    # NEW! Gate 2 metadata fields
    face_confidence_gate2: float = 0.0  # MediaPipe confidence from Gate 2
    snr_gate2: float = 0.0              # Frame SNR from Gate 2
    landmarks_quality: float = 0.0      # Landmarks validation score (0-1)


__all__ = ["EmbeddingResult"]
