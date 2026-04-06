"""Unified exports for FaceGuard settings package."""

from .path import (
    ROOT_DIR,
    CONFIG_DIR,
    DATA_DIR,
    TEMP_DIR,
    DATABASE_DIR,
    DATABASE_SQLITE_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    CACHE_ROOT,
    ensure_core_dirs,
    get_person_temp_dir,
)
from .data_collection import DATA_COLLECTION_CONFIG, get_video_dir
from .frame_extraction import FRAME_EXTRACTION_CONFIG, get_person_frame_dir, get_frame_type_dir
from .embedding import EMBEDDING_CONFIG, EMBEDDING_PATHS, MODEL_URLS, get_person_vector_dir, get_person_metadata_dir
from .sanitizer import SANITIZER_CONFIG, SANITIZER_THRESHOLDS, get_person_split_dir, get_split_type_dir
from .recognition import RECOGNITION_CONFIG, RECOGNITION_QUALITY_THRESHOLDS, get_person_training_input_dir
from .fine_tune import FINE_TUNE_CONFIG
from .frame_sanitizer import FRAME_SANITIZER_CONFIG, get_person_sanitized_dir, get_sanitized_type_dir
from .video_quality import VIDEO_QUALITY_CONFIG, TEMP_PROCESSING_DIR
from .storage import MINIO_STORAGE_CONFIG, CACHE_CONFIG
from .IoT import IOT_SERVICE_CONFIG, IOT_API_CONFIG, IOT_MINIO_KEYS
from .packager import PACKAGER_CONFIG, PACKAGER_RUNTIME_KEYS
from .logging import (
    ERROR,
    CRITICAL,
    FAIL,
    WARNING,
    SKIP,
    SUCCESS,
    PASS,
    SAVE,
    UPLOAD,
    PHASE,
    STAGE,
    STEP,
    INFO,
    METADATA,
    DEBUG,
    TEST,
    LOAD,
    DOWNLOAD,
    console,
    create_table,
    print_section_header,
)


__all__ = [
    # path
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
    # phase configs
    "DATA_COLLECTION_CONFIG",
    "FRAME_EXTRACTION_CONFIG",
    "EMBEDDING_CONFIG",
    "EMBEDDING_PATHS",
    "MODEL_URLS",
    "SANITIZER_CONFIG",
    "SANITIZER_THRESHOLDS",
    "RECOGNITION_CONFIG",
    "RECOGNITION_QUALITY_THRESHOLDS",
    "FINE_TUNE_CONFIG",
    "FRAME_SANITIZER_CONFIG",
    "VIDEO_QUALITY_CONFIG",
    "TEMP_PROCESSING_DIR",
    "PACKAGER_CONFIG",
    "PACKAGER_RUNTIME_KEYS",
    # storage/iot
    "MINIO_STORAGE_CONFIG",
    "CACHE_CONFIG",
    "IOT_SERVICE_CONFIG",
    "IOT_API_CONFIG",
    "IOT_MINIO_KEYS",
    # helpers
    "get_video_dir",
    "get_person_frame_dir",
    "get_frame_type_dir",
    "get_person_vector_dir",
    "get_person_metadata_dir",
    "get_person_split_dir",
    "get_split_type_dir",
    "get_person_training_input_dir",
    "get_person_sanitized_dir",
    "get_sanitized_type_dir",
    # logging labels + helpers
    "ERROR",
    "CRITICAL",
    "FAIL",
    "WARNING",
    "SKIP",
    "SUCCESS",
    "PASS",
    "SAVE",
    "UPLOAD",
    "PHASE",
    "STAGE",
    "STEP",
    "INFO",
    "METADATA",
    "DEBUG",
    "TEST",
    "LOAD",
    "DOWNLOAD",
    "console",
    "create_table",
    "print_section_header",
]
