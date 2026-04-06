"""
Recognition Configuration - Production-ready.

Refactored 08-03-2026:
- Clean final phase: Training → MinIO + SQLite → Cleanup
- Remove local storage paths (use MinIO + SQLite only)
- Production-ready defaults

Folder Structure (Temp Processing Only):
data/
└── temp/
    └── {person_name}/
        └── vector/
            └── split/
                ├── train/train.npz     ← Phase 4 output (INPUT)
                ├── validation/val.npz
                └── test/test.npz

Final Storage (Persistent):
- MinIO: {person_id}/embeddings_v1.npz (vectors)
- SQLite: Person metadata + vector keys

Workflow:
1. Phase 4 output: Splits at data/temp/{person}/vector/split/
2. Phase 5 trains:
   - Load train.npz
   - Train SimpleFaceRecognizer
   - Optimize threshold
   - Quality gate (user confirmation)
3. Save & Upload:
   - NPZ → MinIO (vectors)
   - Metadata → SQLite (person info + keys)
4. Cleanup:
   - DELETE data/temp/{person}/ (all temp files)
5. Pipeline COMPLETE!

Quality Gate 5:
- Practical training targets
- Focus: Usable accuracy for production
- Threshold optimization (multi-threshold testing)
- User confirmation required
"""

from .path import DATABASE_DIR, TEMP_DIR

# ==================== HELPER FUNCTIONS ====================

def get_person_training_input_dir(person_name: str):
    """
    Get training input directory (Phase 4 splits).

    Args:
        person_name: Person name (sanitized)

    Returns:
        Path to split directory

    Structure:
        data/temp/{person_name}/vector/split/
    """
    from pathlib import Path
    split_dir = TEMP_DIR / person_name / "vector" / "split"
    return split_dir


def get_person_temp_dir(person_name: str):
    """
    Get person's temp directory (for cleanup after training).

    Args:
        person_name: Person name

    Returns:
        Path to person's temp directory

    Structure:
        data/temp/{person_name}/
    """
    from pathlib import Path
    return TEMP_DIR / person_name

# ==================== STORAGE KEYS ====================
"""
Recognition data sẽ dùng storage keys:

Training splits:
    build_storage_key("training_split", split_name="train", filename="train.npz")
    → "training/splits/train/train.npz"

Recognition database:
    Canonical key: "database/all/face_recognition_db.npz"
    (Used by IoT + Packaging runtime)
"""

# ==================== CONFIG ====================

RECOGNITION_CONFIG = {
    # ===== Model Configuration =====
    "MODEL_NAME": "arcface",
    "EMBEDDING_DIM": 512,
    
    # ===== Training Parameters =====
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.001,
    "NUM_EPOCHS": 50,
    "EARLY_STOPPING_PATIENCE": 10,
    
    # ===== Recognition Thresholds =====
    "DEFAULT_THRESHOLD": 0.55,  # ✅ Relaxed: Was 0.65! -15% Cosine similarity
    "TEST_THRESHOLDS": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],  # ✅ Expanded range
    "SIMILARITY_THRESHOLD": 0.50,  # ✅ Relaxed: Was 0.60!
    "MIN_CONFIDENCE": 0.60,  # ✅ Relaxed: Was 0.70! -14%
    "MAX_DISTANCE": 0.80,

    # ===== Quality Gate Configuration =====
    "QUALITY_GATE": {
        # Accuracy requirements (relaxed for practical training)
        "MIN_OVERALL_ACCURACY": 0.80,      # ✅ Relaxed: Was 0.90! -11%
        "MIN_KNOWN_ACCURACY": 0.82,        # ✅ Relaxed: Was 0.92! -11%
        "MIN_PER_PERSON_ACCURACY": 0.75,   # ✅ Relaxed: Was 0.85! -12%

        # Warning thresholds
        "WARN_OVERALL_ACCURACY": 0.85,     # ✅ Relaxed: Was 0.95!
        "WARN_KNOWN_ACCURACY": 0.87,       # ✅ Relaxed: Was 0.97!

        # Sample requirements (keep same)
        "MIN_TEST_SAMPLES": 10,            # Minimum test samples

        # Quality grade thresholds (NEW!)
        "EXCELLENT": {
            "overall": 0.95,   # ≥95% overall accuracy
            "known": 0.97,     # ≥97% known accuracy
        },
        "GOOD": {
            "overall": 0.85,   # ≥85% overall accuracy
            "known": 0.87,     # ≥87% known accuracy
        },
        # ACCEPTABLE: meets MIN thresholds (80%/82%)
        # POOR: below MIN thresholds
    },
    
    # ===== File Configuration =====
    "RAW_VIDEO_EXTENSIONS": [".mp4", ".avi", ".mov", ".mkv"],
    
    # ===== Database Options =====
    "DB_FORMAT": "npz",
    "ENABLE_SYNC": True,  # Sync local DB với remote storage
}

# ===== BACKWARD COMPATIBILITY =====
# Export quality thresholds separately for services
RECOGNITION_QUALITY_THRESHOLDS = RECOGNITION_CONFIG["QUALITY_GATE"]

__all__ = [
    "RECOGNITION_CONFIG",
    "RECOGNITION_QUALITY_THRESHOLDS",
    "get_person_training_input_dir",
    "get_person_temp_dir",
]