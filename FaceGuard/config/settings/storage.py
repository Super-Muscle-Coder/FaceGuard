"""Storage configuration (MinIO + local cache)."""

from __future__ import annotations

import os

from .path import CACHE_ROOT


MINIO_STORAGE_CONFIG = {
    "ENDPOINT": os.getenv("FACEGUARD_MINIO_ENDPOINT", "localhost:9000"),
    "ACCESS_KEY": os.getenv("FACEGUARD_MINIO_ACCESS_KEY")
    or os.getenv("MINIO_ROOT_USER")
    or "MyMinIOAccount",
    "SECRET_KEY": os.getenv("FACEGUARD_MINIO_SECRET_KEY")
    or os.getenv("MINIO_ROOT_PASSWORD")
    or "12345678",
    "SECURE": os.getenv("FACEGUARD_MINIO_SECURE", "false").lower() == "true",
    "BUCKET_NAME": os.getenv("FACEGUARD_MINIO_BUCKET", "faceguard"),
}


CACHE_CONFIG = {
    "ENABLE_CACHE": os.getenv("FACEGUARD_ENABLE_CACHE", "true").lower() == "true",
    "CACHE_DIR": CACHE_ROOT / "minio",
}


__all__ = ["MINIO_STORAGE_CONFIG", "CACHE_CONFIG"]
