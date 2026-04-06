"""
Data collection configuration 

Refactored 08-03-2026:
- Clean folder structure: data/temp/{person}/video/{type}/
- Remove test-specific configs
- Production-ready defaults

Folder Structure:
data/
└── temp/
    └── {person_name}/
        └── video/
            ├── frontal/
            ├── horizontal/
            └── vertical/

Workflow:
1. User enters name
2. Create temp folders for person
3. Record 3 videos (frontal, horizontal, vertical)
4. Each video: Record → Quality Check (Gate 1) → Playback → Keep/Reject
5. Videos saved to respective folders
6. Ready for Phase 2 (Frame Extraction)
"""
from .path import TEMP_DIR

# ==================== HELPER FUNCTIONS ====================

def get_person_temp_dir(person_name: str):
    """
    Get temp directory for a person.

    Args:
        person_name: Person name (sanitized)

    Returns:
        Path to person's temp directory

    Structure:
        data/temp/{person_name}/
    """
    from pathlib import Path
    person_dir = TEMP_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir


def get_video_dir(person_name: str, video_type: str):
    """
    Get video directory for specific type.

    Args:
        person_name: Person name
        video_type: Video type (frontal/horizontal/vertical)

    Returns:
        Path to video type directory

    Structure:
        data/temp/{person_name}/video/{video_type}/
    """
    person_dir = get_person_temp_dir(person_name)
    video_dir = person_dir / "video" / video_type
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


# ==================== CONFIG ====================

DATA_COLLECTION_CONFIG = {
    # Recording settings
    "RECORDING_DURATION": 15,  # seconds
    "CAMERA_ID": 0,
    "VIDEO_RESOLUTION": (640, 480),
    "VIDEO_FPS": 30,
    "VIDEO_CODEC": "mp4v",  # mp4v = MPEG-4

    # Quality thresholds
    "MIN_FILE_SIZE_BYTES": 100 * 1024,  # 100 KB minimum
    "MIN_FRAME_COUNT": 10,  # Minimum frames required
    "AUTO_REJECT_THRESHOLD": 0.6,  # Auto-reject if quality < 60%
}

__all__ = [
    "DATA_COLLECTION_CONFIG",
    "get_person_temp_dir",
    "get_video_dir",
]
