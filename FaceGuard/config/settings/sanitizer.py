"""
Sanitizer Configuration - Production-ready.

Refactored 08-03-2026:
- Clean folder structure: data/temp/{person}/vector/split/{train|val|test}/
- Remove test-specific configs
- Production-ready defaults

Folder Structure:
data/
└── temp/
    └── {person_name}/
        └── vector/
            ├── face_embeddings.npz  ← Phase 3 output
            └── split/                ← Phase 4 output
                ├── train/
                │   └── train.npz
                ├── validation/
                │   └── val.npz
                └── test/
                    └── test.npz

Workflow:
1. Phase 3 output: NPZ at data/temp/{person}/vector/face_embeddings.npz
2. Phase 4 sanitizes & splits:
   - Validate integrity (11 checks)
   - Detect outliers
   - Clean data (quality filter)
   - Split stratified (70/15/15)
   - Save to split/ subfolders
3. Ready for Phase 5 (Training)

Quality Gate 4:
- Safety net (catch critical failures only)
- Focus: Embedding integrity + outliers
- Stratified splitting (by person + image_type)
"""

from .path import TEMP_DIR

# ==================== HELPER FUNCTIONS ====================

def get_person_split_dir(person_name: str):
    """
    Get split directory for a person.

    Args:
        person_name: Person name (sanitized)

    Returns:
        Path to person's split directory

    Structure:
        data/temp/{person_name}/vector/split/
    """
    from pathlib import Path
    split_dir = TEMP_DIR / person_name / "vector" / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def get_split_type_dir(person_name: str, split_type: str):
    """
    Get directory for specific split type.

    Args:
        person_name: Person name
        split_type: Split type (train/validation/test)

    Returns:
        Path to split type directory

    Structure:
        data/temp/{person_name}/vector/split/{split_type}/
    """
    split_dir = get_person_split_dir(person_name)
    type_dir = split_dir / split_type
    type_dir.mkdir(parents=True, exist_ok=True)
    return type_dir


# ==================== STORAGE KEYS ====================
# Input embeddings: build_storage_key("embedding_batch", filename="face_embeddings.csv")
# Sanitized data: build_storage_key("sanitized_data", filename="cleaned.npz")
# Splits: build_storage_key("sanitized_split", split_name="train", filename="train.npz")
# Reports: build_storage_key("sanitizer_report", filename="validation_report.json")
# Graphs: build_storage_key("sanitizer_graph", filename="tsne_plot.png")

# ==================== THRESHOLDS ====================

SANITIZER_THRESHOLDS = {
    # ════════════════════════════════════════════════════════
    # DETECTION & QUALITY THRESHOLDS (Safety Net!)
    # ════════════════════════════════════════════════════════

    # Detection threshold: MUCH LOWER than Gate 3 (0.50/0.40/0.35)
    # Gate 4 = final safety net, only catch extreme failures!
    "MIN_DETECTION_SCORE": 0.45,  # Relaxed: Was 0.60! -25% looser

    # Quality threshold: MUCH LOWER than Gate 3 (0.70/0.40)
    # Gate 3 already filtered, Gate 4 just catches edge cases
    "MIN_QUALITY_SCORE": 0.45,  # Relaxed: Was 0.60! -25% looser

    # Outlier detection (keep same)
    "OUTLIER_CONTAMINATION": 0.05,  # 5% expected outliers

    # ════════════════════════════════════════════════════════
    # GATE 2-3 METADATA THRESHOLDS (Relaxed!)
    # ════════════════════════════════════════════════════════

    # Embedding norm validation (looser than Gate 3: 0.92-1.08)
    "EMBEDDING_NORM_MIN": 0.90,  # ✅ Relaxed: Was 0.95! -5% wider
    "EMBEDDING_NORM_MAX": 1.10,  # ✅ Relaxed: Was 1.05! +5% wider

    # Gate 2 MediaPipe confidence (much looser - Gate 2 filtered at 0.4)
    "GATE2_FACE_CONFIDENCE_MIN": 0.35,  # ✅ Relaxed: Was 0.50! -30% looser

    # Gate 2 Frame SNR (much looser - Gate 2 filtered at 5dB)
    "GATE2_SNR_MIN": 5.0,  # ✅ Relaxed: Was 8.0! -37% looser

    # Gate 3 Landmarks quality (looser)
    "LANDMARKS_QUALITY_MIN": 0.20,  # ✅ Relaxed: Was 0.30! -33% looser

    # ════════════════════════════════════════════════════════
    # ENABLE/DISABLE GATE 2-3 CHECKS
    # ════════════════════════════════════════════════════════

    "ENABLE_NORM_CHECK": True,          # Check embedding norm range
    "ENABLE_GATE2_CONFIDENCE_CHECK": False,  # ❌ DISABLED: Gate 2 metadata not available yet!
    "ENABLE_GATE2_SNR_CHECK": False,     # ❌ DISABLED: Gate 2 metadata not available yet!
    "ENABLE_LANDMARKS_CHECK": True,     # Check landmarks quality
}

# ==================== CONFIG ====================

SANITIZER_CONFIG = {
    "BALANCE_STRATEGY": "none",
    "ENABLE_PCA": True,
    "ENABLE_TSNE": True,
    "SAVE_INTERMEDIATE": True,
    "SPLIT_RATIOS": {"train": 0.70, "val": 0.15, "test": 0.15},
    "MULTIFACE_STRATEGY": "keep",
}

__all__ = [
    "SANITIZER_THRESHOLDS",
    "SANITIZER_CONFIG",
    "get_person_split_dir",
    "get_split_type_dir",
]
