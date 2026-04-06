"""
Frame Extraction Configuration - Production-ready.

Refactored 08-03-2026:
- Clean folder structure: data/temp/{person}/frames/{type}/
- Remove test-specific configs
- Production-ready defaults

Folder Structure:
data/
└── temp/
    └── {person_name}/
        └── frames/
            ├── frontal/
            │   ├── frame_0001.jpg
            │   └── metadata.csv
            ├── horizontal/
            │   ├── frame_0001.jpg
            │   └── metadata.csv
            └── vertical/
                ├── frame_0001.jpg
                └── metadata.csv

Workflow:
1. Phase 1 output: Videos at data/temp/{person}/video/{type}/
2. Phase 2 extracts frames to data/temp/{person}/frames/{type}/
3. Metadata CSV saved with frames
4. Ready for Phase 3 (Embedding Generation)

Quality Gate 2:
- Looser than Gate 1 (video level)
- Focus: Extract enough quality frames
- Haar Cascade detection (simple, stable)
- SNR check (align with Gate 1)
- Smart ranking (multi-factor)
"""

from .path import TEMP_DIR

# ==================== HELPER FUNCTIONS ====================

def get_person_frame_dir(person_name: str):
    """
    Get frame directory for a person.

    Args:
        person_name: Person name (sanitized)

    Returns:
        Path to person's frame directory

    Structure:
        data/temp/{person_name}/frames/
    """
    from pathlib import Path
    person_dir = TEMP_DIR / person_name / "frames"
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir


def get_frame_type_dir(person_name: str, frame_type: str):
    """
    Get frame directory for specific type.

    Args:
        person_name: Person name
        frame_type: Frame type (frontal/horizontal/vertical)

    Returns:
        Path to frame type directory

    Structure:
        data/temp/{person_name}/frames/{frame_type}/
    """
    person_frame_dir = get_person_frame_dir(person_name)
    type_dir = person_frame_dir / frame_type
    type_dir.mkdir(parents=True, exist_ok=True)
    return type_dir

# ==================== STORAGE KEYS ====================
"""
Frames output sẽ dùng storage keys:

Frames:
    build_storage_key("frame", 
        person="Lien", 
        video_type="frontal", 
        filename="frame_001.jpg"
    )
    → "Lien/frontal/frames/frame_001.jpg"

Metadata:
    build_storage_key("frame_metadata",
        person="Lien",
        video_type="frontal", 
        filename="metadata.csv"
    )
    → "Lien/frontal/metadata/metadata.csv"
"""

# ==================== CONFIG ====================

FRAME_EXTRACTION_CONFIG = {
    # ════════════════════════════════════════════════════════
    # FACE DETECTION (Haar Cascade - Simple & Reliable)
    # ════════════════════════════════════════════════════════

    "ENABLE_FACE_DETECTION": True,
    "MIN_FACE_SIZE": 80,  # ✅ UPDATED: Industry standard (was 40px - too loose!)
    "REQUIRE_FACE_FOR_EXTRACTION": False,  # Allow frames without faces (loose gate 2)

    # ════════════════════════════════════════════════════════
    # QUALITY CHECKS
    # ════════════════════════════════════════════════════════

    "ENABLE_EXPOSURE_CHECK": False,  # ✅ DISABLED: Indoor lighting causes false overexposure
    "SAVE_METADATA": True,

    # Exposure: For future use if re-enabled (more relaxed for indoor)
    "EXPOSURE_CLIP_THRESHOLD": 0.30,  # ✅ Relaxed: Was 0.12! +150% more lenient for indoor

    # SNR: DISABLED (consumer cameras have inconsistent SNR - rely on blur/brightness only)
    "ENABLE_FRAME_SNR_CHECK": False,  # ✅ DISABLED: Consumer webcams have low SNR (4-5 dB)
    "FRAME_MIN_SNR": 3.0,  # Not used when disabled

    # ════════════════════════════════════════════════════════
    # BLUR THRESHOLDS (Looser than Gate 1!)
    # ════════════════════════════════════════════════════════

    # Gate 1: 25, Gate 2: 20 (looser to extract more frames)
    "FRONTAL_BLUR_THRESHOLD": 20,  # ✅ Relaxed: Was 35! -43% less strict
    "HORIZONTAL_BLUR_THRESHOLD": 20,  # ✅ Relaxed: Was 35! -43% less strict
    "VERTICAL_BLUR_THRESHOLD": 18,  # ✅ Relaxed: Was 30! -40% less strict

    # ════════════════════════════════════════════════════════
    # BRIGHTNESS RANGES (Keep same - already reasonable)
    # ════════════════════════════════════════════════════════

    "FRONTAL_BRIGHTNESS_RANGE": (50, 220),
    "HORIZONTAL_BRIGHTNESS_RANGE": (40, 220),
    "VERTICAL_BRIGHTNESS_RANGE": (40, 220),

    # ════════════════════════════════════════════════════════
    # SAMPLING STRATEGIES
    # ════════════════════════════════════════════════════════

    "HORIZONTAL_MAX_PER_BUCKET": 20,
    "VERTICAL_MAX_PER_BUCKET": 20,
    "HORIZONTAL_ANGLE_BOUNDS": (-45, -15, 15, 45),
    "VERTICAL_PROGRESS_SPLITS": (0.3, 0.45, 0.6, 0.8),

    # Frame selection
    "SAMPLE_INTERVAL_MIN": 3,
    "SAMPLE_INTERVAL_DENOM": 200,
    "MIN_SELECT_INTERVAL_BASE": 5,

    # ════════════════════════════════════════════════════════
    # SMART RANKING (NEW! Multi-factor quality score)
    # ════════════════════════════════════════════════════════

    "ENABLE_SMART_RANKING": True,
    "RANKING_WEIGHTS": {
        'blur': 0.30,       # Blur score (reduced from 100%)
        'face': 0.40,       # Face quality (confidence × size) - Most important!
        'exposure': 0.15,   # Exposure quality
        'snr': 0.15,        # Signal-to-Noise Ratio
    },

    # ════════════════════════════════════════════════════════
    # ADVANCED FEATURES (Optional optimizations)
    # ════════════════════════════════════════════════════════

    "ENABLE_TEMPORAL_CONSISTENCY": False,  # Disabled for speed
    "ENABLE_OUTLIER_REMOVAL": False,       # Let Sanitizer handle
    "ENABLE_EMBEDDING_PRECHECK": False,    # Disabled for separation of concerns
}

__all__ = [
    "FRAME_EXTRACTION_CONFIG",
    "get_person_frame_dir",
    "get_frame_type_dir",
]