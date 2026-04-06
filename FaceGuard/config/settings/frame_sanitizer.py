"""Frame sanitizer configuration (Phase 3 for fine-tune research pipeline)."""

from __future__ import annotations

from .path import TEMP_DIR


def get_person_sanitized_dir(person_name: str):
    """Get sanitized frame root directory for a person."""
    person_dir = TEMP_DIR / person_name / "sanitized_frames"
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir


def get_sanitized_type_dir(person_name: str, frame_type: str):
    """Get sanitized frame directory for a specific angle/type."""
    out = get_person_sanitized_dir(person_name) / frame_type
    out.mkdir(parents=True, exist_ok=True)
    return out


FRAME_SANITIZER_CONFIG = {
    "ENABLED": True,
    "ANGLE_TYPES": ["frontal", "horizontal", "vertical"],
    "IMAGE_EXTS": [".jpg", ".jpeg", ".png"],
    "ENABLE_FACE_CHECK": True,
    "ENABLE_ALIGN_FACE": True,
    "ARCFACE_INPUT_SIZE": (112, 112),
    "FACE_MARGIN_RATIO": 0.15,
    "MIN_FACE_SIZE": 80,
    "MIN_QUALITY_SCORE": 0.45,
    "MAX_KEEP_PER_ANGLE": 120,
    "MIN_KEEP_PER_ANGLE": 20,
    "OUTPUT_JPEG_QUALITY": 95,
    "FRONTAL_BLUR_THRESHOLD": 60.0,
    "HORIZONTAL_BLUR_THRESHOLD": 55.0,
    "VERTICAL_BLUR_THRESHOLD": 50.0,
    "FRONTAL_BRIGHTNESS_RANGE": (50.0, 220.0),
    "HORIZONTAL_BRIGHTNESS_RANGE": (40.0, 220.0),
    "VERTICAL_BRIGHTNESS_RANGE": (40.0, 220.0),
}


__all__ = [
    "FRAME_SANITIZER_CONFIG",
    "get_person_sanitized_dir",
    "get_sanitized_type_dir",
]
