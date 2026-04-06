"""IoT runtime configuration for ESP32CAM -> FaceGuard server pipeline."""

from __future__ import annotations

import os


IOT_SERVICE_CONFIG = {
    "HOST": "0.0.0.0",
    "PORT": int(os.getenv("FACEGUARD_IOT_PORT", "5000")),
    "DEBUG": False,
    "REQUEST_TIMEOUT_SECONDS": 10,
    "MAX_IMAGE_SIZE_MB": 5,
    "MAX_CAMERAS": 32,
    "CAMERA_STALE_SECONDS": 120,
    "DEFAULT_RECOGNITION_THRESHOLD": float(os.getenv("FACEGUARD_IOT_THRESHOLD", "0.40")),
    "USE_FINETUNE_HEAD": True,
    "FINETUNE_HEAD_BLEND_ALPHA": 0.35,
    "FINETUNE_HEAD_OVERRIDE_CONF": 0.80,
    "FACE_CROP_MARGIN_RATIO": 0.15,
    "ENABLE_CLAHE_ENHANCE": True,
    "STREAM_JPEG_QUALITY": 80,
    "STREAM_STALE_SECONDS": 10,
    "API_KEY_HEADER": os.getenv("FACEGUARD_API_KEY_HEADER", "X-API-Key"),
    "API_KEY": os.getenv("FACEGUARD_API_KEY", "FaceGuard-IoT-ESP32CAM-Key-ChangeMe"),
}


IOT_API_CONFIG = {
    "HEALTH_PATH": "/health",
    "METRICS_PATH": "/metrics",
    "CAMERAS_PATH": "/cameras",
    "RELOAD_PATH": "/reload",
    "RECOGNIZE_PATH": "/recognize",
    "API_V1_PREFIX": "/api/v1",
}


IOT_MINIO_KEYS = {
    "DATABASE_NPZ": "database/all/face_recognition_db.npz",
    "EVENTS_PREFIX": "iot/events",
    "SNAPSHOTS_PREFIX": "iot/snapshots",
}


__all__ = ["IOT_SERVICE_CONFIG", "IOT_API_CONFIG", "IOT_MINIO_KEYS"]
