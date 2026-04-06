from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FineTuneEpochMetrics:
    """Per-epoch training and validation metrics."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    score: float = 0.0
    per_class_val_acc: Dict[str, float] = field(default_factory=dict)


@dataclass
class FineTuneReport:
    """Final report for fine-tune run."""
    person_name: str
    epochs: int
    best_val_acc: float
    best_epoch: int
    best_score: float = 0.0
    early_stopped: bool = False
    stopped_epoch: int = 0
    metrics: List[FineTuneEpochMetrics] = field(default_factory=list)
    class_to_index: Dict[str, int] = field(default_factory=dict)


__all__ = ["FineTuneEpochMetrics", "FineTuneReport"]
