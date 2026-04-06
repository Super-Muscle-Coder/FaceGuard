"""Realtime packaging/runtime test configuration (Phase 6)."""

from __future__ import annotations

from .IoT import IOT_MINIO_KEYS


PACKAGER_CONFIG = {
    # Camera / runtime
    "CAMERA_ID": 0,
    "WINDOW_WIDTH": 1280,
    "WINDOW_HEIGHT": 720,
    "CAMERA_WIDTH": 640,
    "CAMERA_HEIGHT": 480,

    # Recognition
    "DEFAULT_THRESHOLD": 0.55,
    "PROCESS_EVERY_N_FRAMES": 2,
    "MAX_FACES_PER_FRAME": 3,
    "MIN_FACE_SIZE": (80, 80),
    "ALLOW_LOCAL_DB_FALLBACK": False,

    # Hybrid recognition (cosine + fine-tune head)
    "USE_FINETUNE_HEAD": True,
    "FINETUNE_HEAD_BLEND_ALPHA": 0.35,  # final = (1-a)*cosine + a*head_prob (when same class)
    "FINETUNE_HEAD_OVERRIDE_CONF": 0.80,  # allow head to override cosine when very confident

    # Tracker
    "TRACKER_IOU_THRESHOLD": 0.30,
    "TRACKER_SMOOTHING_FACTOR": 0.70,

    # Visual
    "KNOWN_COLOR": (0, 255, 0),
    "UNKNOWN_COLOR": (0, 0, 255),
    "BOX_THICKNESS": 2,
    "FONT_SCALE": 0.6,
}


PACKAGER_RUNTIME_KEYS = {
    "DATABASE_NPZ": IOT_MINIO_KEYS["DATABASE_NPZ"],
}


__all__ = ["PACKAGER_CONFIG", "PACKAGER_RUNTIME_KEYS"]

