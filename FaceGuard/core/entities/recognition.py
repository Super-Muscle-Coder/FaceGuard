from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class RecognitionResult:
    """Kết quả nhận diện cho 1 ảnh."""
    predicted_label: str
    true_label: str
    confidence: float
    similarities: Dict[str, float]
    is_correct: bool
    is_known: bool


@dataclass
class EvaluationMetrics:
    """Metrics đánh giá hệ thống nhận diện."""
    total_samples: int
    accuracy: float
    known_samples: int
    known_correct: int
    known_accuracy: float
    per_person_accuracy: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


__all__ = ["RecognitionResult", "EvaluationMetrics"]
