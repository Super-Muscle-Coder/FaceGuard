from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import numpy as np


@dataclass
class ValidationReport:
    """
    Report from validation phase.

    Refactored 31-01-2026:
    - Add Gate 2-3 metadata stats (embedding_norm, face_confidence, SNR, landmarks)
    - Enhanced reporting for multi-gate quality tracking
    """
    passed: bool
    has_nan: bool
    has_inf: bool
    is_normalized: bool
    has_duplicates: bool
    label_counts: Dict[str, int]
    type_counts: Dict[str, int]
    quality_stats: Dict[str, float]
    outlier_mask: Optional[np.ndarray] = None

    # ════════════════════════════════════════════════════════
    # NEW! Gate 2-3 Metadata Statistics
    # ════════════════════════════════════════════════════════

    embedding_norm_stats: Dict[str, float] = field(default_factory=dict)
    # {'mean': 1.0, 'std': 0.01, 'min': 0.98, 'max': 1.02, 'out_of_range_count': 5}

    gate2_confidence_stats: Dict[str, float] = field(default_factory=dict)
    # {'mean': 0.85, 'std': 0.10, 'min': 0.60, 'max': 0.98, 'low_count': 3}

    gate2_snr_stats: Dict[str, float] = field(default_factory=dict)
    # {'mean': 14.5, 'std': 2.1, 'min': 9.0, 'max': 20.0, 'low_count': 2}

    landmarks_quality_stats: Dict[str, float] = field(default_factory=dict)
    # {'mean': 0.85, 'std': 0.20, 'min': 0.40, 'max': 1.0, 'low_count': 4}

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CleaningReport:
    """Report to cleaning phase."""
    original_size: int
    cleaned_size: int
    removed_count: int
    removal_rate: float
    removed_by_quality: int
    removed_by_outliers: int
    removed_by_multiface: int
    removed_by_balance: int
    final_label_counts: Dict[str, int]
    final_type_counts: Dict[str, int]
    final_quality_stats: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SplitReport:
    """Report to splitting phase."""
    total_samples: int
    train_size: int
    val_size: int
    test_size: int
    
    # Distribution per split
    train_distribution: Dict[str, int]
    val_distribution: Dict[str, int]
    test_distribution: Dict[str, int]
    
    # Distribution per image_type
    train_type_dist: Dict[str, int]
    val_type_dist: Dict[str, int]
    test_type_dist: Dict[str, int]
    
    # Stratification quality (1.0 = perfect balance)
    stratification_score: float
    
    # Split ratios
    train_ratio: float
    val_ratio: float
    test_ratio: float
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


__all__ = ["ValidationReport", "CleaningReport", "SplitReport"]
