"""Central path configuration for FaceGuard."""

from __future__ import annotations

from pathlib import Path


# ==================== ROOT ====================

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"


# ==================== DATA LAYOUT ====================

DATA_DIR = ROOT_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"

DATABASE_DIR = ROOT_DIR / "database"
DATABASE_SQLITE_PATH = DATABASE_DIR / "face_recognition.db"


# ==================== MODELS ====================

MODELS_DIR = ROOT_DIR / "scripts" / "models"


# ==================== REPORTS/CACHE ====================

REPORTS_DIR = ROOT_DIR / "report"
CACHE_ROOT = ROOT_DIR / ".cache"


def ensure_core_dirs() -> None:
    """Ensure core project directories exist."""
    for path in [
        DATA_DIR,
        TEMP_DIR,
        DATABASE_DIR,
        CACHE_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_person_temp_dir(person_name: str) -> Path:
    """Get temp working directory for one person."""
    person_dir = TEMP_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir


ensure_core_dirs()


__all__ = [
    "ROOT_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "TEMP_DIR",
    "DATABASE_DIR",
    "DATABASE_SQLITE_PATH",
    "MODELS_DIR",
    "REPORTS_DIR",
    "CACHE_ROOT",
    "ensure_core_dirs",
    "get_person_temp_dir",
]
