"""
Embedding Configuration - Production-ready.

Refactored 08-03-2026:
- Clean folder structure: data/temp/{person}/vector/ and /metadata/
- Remove test-specific configs
- Production-ready defaults

Folder Structure:
data/
└── temp/
    └── {person_name}/
        ├── vector/
        │   └── face_embeddings.npz  ← NPZ vectors for Phase 4
        └── metadata/
            └── face_embeddings_metadata.json  ← JSON for SQLite (Phase 6)

Workflow:
1. Phase 2 output: Frames at data/temp/{person}/frames/{type}/
2. Phase 3 extracts embeddings:
   - NPZ vectors → data/temp/{person}/vector/
   - JSON metadata → data/temp/{person}/metadata/
3. NPZ ready for Phase 4 (Sanitization)
4. JSON ready for Phase 6 (Database after training)

Quality Gate 3:
- Looser than Gate 2 (frame level)
- Focus: Generate embeddings from quality frames
- SCRFD detection (95% accuracy)
- ArcFace embeddings (512-dim L2-normalized)
- Quality scoring (detection + landmarks + embedding norm)
"""

from .path import MODELS_DIR, TEMP_DIR

# ==================== HELPER FUNCTIONS ====================

def get_person_vector_dir(person_name: str):
    """
    Get vector directory for a person.

    Args:
        person_name: Person name (sanitized)

    Returns:
        Path to person's vector directory

    Structure:
        data/temp/{person_name}/vector/
    """
    from pathlib import Path
    person_dir = TEMP_DIR / person_name / "vector"
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir


def get_person_metadata_dir(person_name: str):
    """
    Get metadata directory for a person.

    Args:
        person_name: Person name (sanitized)

    Returns:
        Path to person's metadata directory

    Structure:
        data/temp/{person_name}/metadata/
    """
    from pathlib import Path
    person_dir = TEMP_DIR / person_name / "metadata"
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir

# ==================== STORAGE KEYS ====================
"""
Embeddings output sẽ dùng storage keys:

CSV:
    build_storage_key("embedding_batch", filename="face_embeddings.csv")
    → "embeddings/face_embeddings.csv"

NPZ:
    build_storage_key("embedding_batch", filename="face_embeddings.npz")
    → "embeddings/face_embeddings.npz"

Parquet:
    build_storage_key("embedding_batch", filename="face_embeddings.parquet")
    → "embeddings/face_embeddings.parquet"
"""



# ==================== MODEL PATHS ====================
# Models ONNX stored locally (not changed)

EMBEDDING_PATHS = {
    "MODELS_DIR": MODELS_DIR,
    "SCRFD_MODEL": MODELS_DIR / "scrfd_10g_bnkps.onnx",
    "ARCFACE_MODEL": MODELS_DIR / "glintr100.onnx",
    "OUTPUT_DIR": TEMP_DIR / "embedding_output",
}

# ==================== MODEL URLS ====================
# URLs to download models if not present

MODEL_URLS = {
    "SCRFD": [
        "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx",
    ],
    "GLINTR100": [
        "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/glintr100.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/glintr100.onnx",
    ],
}

# ==================== CONFIG ====================

EMBEDDING_CONFIG = {
    # ════════════════════════════════════════════════════════
    # DETECTION THRESHOLDS (Looser than Gate 2!)
    # ════════════════════════════════════════════════════════

    # Gate 2 MediaPipe: 0.4, Gate 3 SCRFD: 0.50 (looser to accept more faces)
    # Rationale: Gate 2 already filtered, Gate 3 generates embeddings permissively

    "DETECTION_THRESHOLD": 0.50,  # ✅ Relaxed: Was 0.65! -23% looser (frontal)
    "NMS_THRESHOLD": 0.40,
    "HORIZONTAL_DETECTION_THRESHOLD": 0.40,  # ✅ Relaxed: Was 0.55! -27% looser
    "VERTICAL_DETECTION_THRESHOLD": 0.35,    # ✅ Relaxed: Was 0.50! -30% looser

    # ════════════════════════════════════════════════════════
    # QUALITY ASSESSMENT (USE GATE 2 METADATA!)
    # ════════════════════════════════════════════════════════

    "QUALITY_THRESHOLD_HIGH": 0.70,  # ✅ Relaxed: Was 0.80! -12% looser
    "QUALITY_THRESHOLD_LOW": 0.40,   # ✅ Relaxed: Was 0.50! -20% looser
    "ENABLE_QUALITY_SCORE": True,

    # UPDATED weights: Focus on embedding quality, not image quality (Gate 2 done!)
    "QUALITY_WEIGHTS": {
        'detection': 0.30,          # SCRFD detection score
        'face_confidence': 0.30,    # MediaPipe confidence from Gate 2
        'snr': 0.20,                # Frame SNR from Gate 2
        'landmarks_quality': 0.10,  # Landmarks validation
        'embedding_norm': 0.10,     # Embedding L2 norm check
    },

    # Legacy params (for backward compat if metadata missing)
    "QUALITY_BLUR_MAX": 200.0,
    "QUALITY_BRIGHTNESS_PEAK": 128.0,
    "QUALITY_EXPOSURE_VALUES": {"good": 1.0, "bad": 0.5},

    # ════════════════════════════════════════════════════════
    # EMBEDDING QUALITY VALIDATION (Relaxed!)
    # ════════════════════════════════════════════════════════

    # Embedding norm range (L2-normalized ArcFace should be ~1.0)
    "EMBEDDING_NORM_MIN": 0.92,  # ✅ Relaxed: Was 0.95! +3% wider range
    "EMBEDDING_NORM_MAX": 1.08,  # ✅ Relaxed: Was 1.05! +3% wider range

    # Landmarks quality (eye distance minimum)
    "MIN_LANDMARKS_DISTANCE": 15,  # ✅ Relaxed: Was 20! -25% smaller allowed

    # ════════════════════════════════════════════════════════
    # PROCESSING OPTIONS
    # ════════════════════════════════════════════════════════

    "ENABLE_PREPROCESSING": True,
    "ENABLE_SHARPENING": True,
    "SAVE_DEBUG_IMAGES": False,

    # ════════════════════════════════════════════════════════
    # RETRY STRATEGIES (Keep same)
    # ════════════════════════════════════════════════════════

    "RETRY_ON_FAILURE": True,
    "RETRY_STRATEGIES": [
        {"label": "Lower threshold", "threshold": 0.30, "preprocess": True},
        {"label": "Minimal threshold", "threshold": 0.20, "preprocess": True},
    ],

    # ════════════════════════════════════════════════════════
    # EXPORT FORMATS
    # ════════════════════════════════════════════════════════

    "ENABLE_PARQUET": True,

    # ════════════════════════════════════════════════════════
    # REMOVED: Blur thresholds (Gate 2 already filtered!)
    # ════════════════════════════════════════════════════════

    # DELETED: "FRONTAL_BLUR_THRESHOLD": 80.0
    # DELETED: "HORIZONTAL_BLUR_THRESHOLD": 80.0
    # DELETED: "VERTICAL_BLUR_THRESHOLD": 70.0
    # Reason: Gate 2 with blur >= 50/50/45 already ensures good blur
    #         No need to check again here!
}

__all__ = [
    "EMBEDDING_PATHS",
    "MODEL_URLS",
    "EMBEDDING_CONFIG",
    "get_person_vector_dir",
    "get_person_metadata_dir",
]