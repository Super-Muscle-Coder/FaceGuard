"""
StorageAdapter - Direct MinIO Integration (SQLite + MinIO Architecture).

FINAL ARCHITECTURE:
===================
💾 SQLite (Metadata):
   - Person info, access logs
   - See: core.storage.SQLite
   - Query: <1ms

🪣 MinIO (Vectors):
   - Face embeddings (NPZ files)
   - Direct MinIO SDK integration
   - Upload/download: ~50ms

❌ NO persistent local storage:
   - Only temp files during processing
   - Temp cache for performance

FEATURES:
- Direct MinIO client (no abstract backend)
- Local file-based cache (optional)
- Backward compatible legacy API
- Simple, clean code

===================  LỊCH SỬ  ====================
Refactored 27-02-2026:
- REMOVED: Abstract backend classes (StorageBackend, MinIOStorageBackend, LocalStorageBackend)
- CHANGED: Direct MinIO SDK integration
- SIMPLIFIED: No abstraction layer
- KEPT: Legacy API (backward compatible)
- FOCUS: SQLite (metadata) + MinIO (vectors) only
"""

import io
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from minio import Minio
from minio.error import S3Error

from config.settings import (
    EMBEDDING_CONFIG,
    EMBEDDING_PATHS,
    TEMP_DIR,  # ✅ Use TEMP_DIR instead of SANITIZER_LOCAL
)
from config.settings.storage import (
    MINIO_STORAGE_CONFIG,
    CACHE_CONFIG,
)
from core.entities.embedding import EmbeddingResult

logger = logging.getLogger(__name__)


class StorageAdapter:
    """
    Storage adapter with direct MinIO integration.

    Architecture:
    - MinIO: Direct SDK client (vectors storage)
    - Local cache: File-based cache (performance)
    - Legacy API: Backward compatible (temp files)

    Usage:
        storage = StorageAdapter()

        # Upload/download vectors
        storage.put("person_id/embeddings.npz", data)
        data = storage.get("person_id/embeddings.npz")

        # File operations
        storage.put_file(local_path, "person_id/embeddings.npz")
        storage.get_file("person_id/embeddings.npz", local_path)
    """

    def __init__(self, minio_client: Optional[Minio] = None):
        """
        Initialize storage adapter.

        Args:
            minio_client: Custom MinIO client (for testing)
                         If None, create from config
        """
        # Setup MinIO client
        if minio_client is not None:
            self.minio = minio_client
        else:
            self.minio = self._create_minio_client()

        self.bucket_name = MINIO_STORAGE_CONFIG["BUCKET_NAME"]

        # Ensure bucket exists
        self._ensure_bucket_exists()

        # Setup cache
        self.cache_enabled = CACHE_CONFIG.get("ENABLE_CACHE", False)
        if self.cache_enabled:
            self._setup_cache()

        logger.info(f"StorageAdapter initialized: MinIO ({self.bucket_name})")

    def _create_minio_client(self) -> Minio:
        """Create MinIO client from config."""
        config = MINIO_STORAGE_CONFIG

        try:
            client = Minio(
                endpoint=config["ENDPOINT"],
                access_key=config["ACCESS_KEY"],
                secret_key=config["SECRET_KEY"],
                secure=config["SECURE"]
            )

            logger.info(f"MinIO client created: {config['ENDPOINT']}")
            return client

        except Exception as ex:
            logger.error(f"Failed to create MinIO client: {ex}")
            raise

    def _ensure_bucket_exists(self):
        """Ensure MinIO bucket exists."""
        try:
            if not self.minio.bucket_exists(self.bucket_name):
                self.minio.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.debug(f"Bucket exists: {self.bucket_name}")

        except S3Error as ex:
            logger.error(f"Failed to ensure bucket exists: {ex}")
            raise

    def _setup_cache(self):
        """Setup local file-based cache."""
        self.cache_dir = CACHE_CONFIG["CACHE_DIR"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache enabled: {self.cache_dir}")

    def _get_cache_path(self, key: str) -> Path:
        """Get local cache file path for a storage key."""
        # Replace slashes with underscores for safe filename
        safe_key = key.replace('/', '_')
        return self.cache_dir / safe_key

    # ==================== CORE STORAGE API ====================

    def put(self, key: str, data: bytes, metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload data to MinIO.

        Args:
            key: Storage key (S3-style path)
            data: Binary data
            metadata: Optional metadata

        Returns:
            True if successful
        """
        try:
            # Upload to MinIO
            data_stream = io.BytesIO(data)

            self.minio.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=data_stream,
                length=len(data),
                metadata=metadata or {}
            )

            logger.debug(f"Uploaded to MinIO: {key} ({len(data)} bytes)")

            # Update cache
            if self.cache_enabled:
                cache_path = self._get_cache_path(key)
                cache_path.write_bytes(data)
                logger.debug(f"Cached: {key}")

            return True

        except S3Error as ex:
            logger.error(f"Failed to upload to MinIO: {key} - {ex}")
            return False
        except Exception as ex:
            logger.error(f"Unexpected error during upload: {key} - {ex}")
            return False

    def get(self, key: str, use_cache: bool = True) -> Optional[bytes]:
        """
        Download data from MinIO.

        Args:
            key: Storage key
            use_cache: Try cache first?

        Returns:
            Data bytes or None
        """
        # Try cache first
        if self.cache_enabled and use_cache:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                try:
                    data = cache_path.read_bytes()
                    logger.debug(f"Cache hit: {key}")
                    return data
                except Exception as ex:
                    logger.warning(f"Cache read failed: {key} - {ex}")

        # Fetch from MinIO
        try:
            response = self.minio.get_object(self.bucket_name, key)
            data = response.read()
            response.close()
            response.release_conn()

            logger.debug(f"Downloaded from MinIO: {key} ({len(data)} bytes)")

            # Update cache
            if self.cache_enabled:
                cache_path = self._get_cache_path(key)
                try:
                    cache_path.write_bytes(data)
                    logger.debug(f"Cached: {key}")
                except Exception as ex:
                    logger.warning(f"Cache write failed: {key} - {ex}")

            return data

        except S3Error as ex:
            if ex.code == 'NoSuchKey':
                logger.debug(f"Key not found: {key}")
            else:
                logger.error(f"Failed to download from MinIO: {key} - {ex}")
            return None
        except Exception as ex:
            logger.error(f"Unexpected error during download: {key} - {ex}")
            return None

    def exists(self, key: str) -> bool:
        """
        Check if key exists in MinIO.

        Args:
            key: Storage key

        Returns:
            True if exists
        """
        # Check cache first
        if self.cache_enabled:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                return True

        # Check MinIO
        try:
            self.minio.stat_object(self.bucket_name, key)
            return True
        except S3Error as ex:
            if ex.code == 'NoSuchKey':
                return False
            else:
                logger.error(f"Error checking existence: {key} - {ex}")
                return False
        except Exception as ex:
            logger.error(f"Unexpected error checking existence: {key} - {ex}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete from MinIO and cache.

        Args:
            key: Storage key

        Returns:
            True if successful
        """
        success = True

        # Delete from MinIO
        try:
            self.minio.remove_object(self.bucket_name, key)
            logger.debug(f"Deleted from MinIO: {key}")
        except S3Error as ex:
            logger.error(f"Failed to delete from MinIO: {key} - {ex}")
            success = False

        # Delete from cache
        if self.cache_enabled:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.debug(f"Deleted from cache: {key}")
                except Exception as ex:
                    logger.warning(f"Cache delete failed: {key} - {ex}")

        return success

    def list(self, prefix: str = "", max_keys: int = 1000) -> List[str]:
        """
        List keys with prefix.

        Args:
            prefix: Key prefix to filter
            max_keys: Maximum number of keys to return

        Returns:
            List of storage keys
        """
        try:
            objects = self.minio.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True
            )

            keys = [obj.object_name for obj in objects]

            # Limit results
            if len(keys) > max_keys:
                keys = keys[:max_keys]
                logger.warning(f"Truncated results to {max_keys} keys")

            logger.debug(f"Listed {len(keys)} keys with prefix: {prefix}")
            return keys

        except S3Error as ex:
            logger.error(f"Failed to list objects: {ex}")
            return []
        except Exception as ex:
            logger.error(f"Unexpected error listing objects: {ex}")
            return []

    def get_url(self, key: str, expiry_seconds: int = 3600) -> Optional[str]:
        """
        Generate temporary access URL.

        Args:
            key: Storage key
            expiry_seconds: URL expiry time

        Returns:
            Presigned URL or None
        """
        try:
            from datetime import timedelta

            url = self.minio.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=key,
                expires=timedelta(seconds=expiry_seconds)
            )

            logger.debug(f"Generated presigned URL: {key}")
            return url

        except S3Error as ex:
            logger.error(f"Failed to generate URL: {key} - {ex}")
            return None
        except Exception as ex:
            logger.error(f"Unexpected error generating URL: {key} - {ex}")
            return None

    # ==================== FILE OPERATIONS ====================

    def put_file(self, local_path: Path, storage_key: str, 
                 metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload file to MinIO.

        Args:
            local_path: Local file path (source)
            storage_key: Storage key (destination)
            metadata: Optional metadata

        Returns:
            True if successful
        """
        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return False

        try:
            data = local_path.read_bytes()
            return self.put(storage_key, data, metadata)
        except Exception as ex:
            logger.error(f"put_file failed: {ex}")
            return False

    def get_file(self, storage_key: str, local_path: Path, use_cache: bool = True) -> bool:
        """
        Download file from MinIO.

        Args:
            storage_key: Storage key (source)
            local_path: Local file path (destination)
            use_cache: Try cache first?

        Returns:
            True if successful
        """
        try:
            data = self.get(storage_key, use_cache)
            if data is None:
                logger.error(f"Failed to download: {storage_key}")
                return False

            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)

            logger.debug(f"Downloaded to: {local_path}")
            return True

        except Exception as ex:
            logger.error(f"get_file failed: {ex}")
            return False

    # ==================== TEMP FILE UTILITIES ====================
    # Backward compatible API for temp file operations during processing

    @staticmethod
    def ensure_dir(path: Path) -> Path:
        """Ensure directory exists (for temp files during processing)."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def move_file(src: Path, dst: Path) -> bool:
        """Move temp file (local filesystem only)."""
        try:
            import shutil
            StorageAdapter.ensure_dir(dst.parent)
            shutil.move(str(src), str(dst))
            logger.debug(f"Moved: {src} -> {dst}")
            return True
        except Exception as ex:
            logger.error(f"move_file failed: {ex}")
            return False

    @staticmethod
    def copy_file(src: Path, dst: Path, overwrite: bool = False) -> bool:
        """Copy temp file (local filesystem only)."""
        try:
            import shutil
            StorageAdapter.ensure_dir(dst.parent)
            if dst.exists() and not overwrite:
                logger.debug(f"Skip copy (exists): {dst}")
                return True
            shutil.copy2(str(src), str(dst))
            logger.debug(f"Copied: {src} -> {dst}")
            return True
        except Exception as ex:
            logger.error(f"copy_file failed: {ex}")
            return False

    # ==================== NPZ OPERATIONS (Temp Processing) ====================
    # These methods work with temp local files during processing

    @staticmethod
    def save_embeddings_npz(npz_path: Path, embeddings: np.ndarray, 
                           labels: np.ndarray, image_paths: np.ndarray) -> bool:
        """Save embeddings to temp NPZ file."""
        try:
            StorageAdapter.ensure_dir(npz_path.parent)
            np.savez_compressed(npz_path, embeddings=embeddings, labels=labels, image_paths=image_paths)
            logger.debug(f"Saved NPZ: {npz_path} shape={embeddings.shape}")
            return True
        except Exception as ex:
            logger.error(f"save_embeddings_npz failed: {ex}")
            return False

    @staticmethod
    def load_npz(npz_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Load NPZ file from local filesystem."""
        if not npz_path.exists():
            logger.error(f"NPZ not found: {npz_path}")
            return None
        try:
            data = np.load(npz_path, allow_pickle=False)
            return {k: data[k] for k in data.files}
        except Exception as ex:
            logger.error(f"load_npz failed: {ex}")
            return None

    @staticmethod
    def save_embeddings_csv(csv_path: Path, rows: Iterable[Dict]) -> bool:
        """Save embeddings to temp CSV file."""
        try:
            StorageAdapter.ensure_dir(csv_path.parent)
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.debug(f"Saved CSV: {csv_path} (rows={len(df)})")
            return True
        except Exception as ex:
            logger.error(f"save_embeddings_csv failed: {ex}")
            return False

    @staticmethod
    def save_parquet(parquet_path: Path, data: Dict) -> bool:
        """Save data to temp Parquet file."""
        try:
            StorageAdapter.ensure_dir(parquet_path.parent)
            df = pd.DataFrame(data)
            df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
            logger.debug(f"Saved Parquet: {parquet_path} rows={len(df)}")
            return True
        except ImportError:
            logger.warning(f"pyarrow not installed, skip parquet: {parquet_path}")
            return False
        except Exception as ex:
            logger.error(f"save_parquet failed: {ex}")
            return False

    # ==================== EMBEDDING BUNDLES (Temp Processing) ====================
    # Save embedding results to temp files (for backward compatibility)

    @staticmethod
    def save_embedding_results(
        results: Iterable[EmbeddingResult],
        output_dir: Optional[Path] = None,
        enable_parquet: Optional[bool] = None,
    ) -> Dict[str, Optional[Path]]:
        """
        Save embedding results to temp files (CSV/NPZ/Parquet).

        Note: Temp files for processing, not persistent storage.
              Use PersonService for persistent metadata + vectors.

        Args:
            results: Iterable of EmbeddingResult
            output_dir: Output directory (default: from config)
            enable_parquet: Enable Parquet output (default: from config)

        Returns:
            Dict with paths: {"csv": Path, "npz": Path, "parquet": Path or None}
        """
        output_dir = output_dir or EMBEDDING_PATHS["OUTPUT_DIR"]
        enable_parquet = (
            EMBEDDING_CONFIG["ENABLE_PARQUET"]
            if enable_parquet is None
            else enable_parquet
        )
        StorageAdapter.ensure_dir(output_dir)

        # Prepare data
        rows: List[Dict] = []
        embeddings: List[np.ndarray] = []
        labels: List[str] = []
        image_paths: List[str] = []
        parquet_rows: Dict[str, List] = {
            "person_name": [],
            "image_type": [],
            "embedding": [],
            "detection_score": [],
            "quality_score": [],
            "embedding_norm": [],
            "image_path": [],
        }

        # Process results
        for r in results:
            emb = r.embedding if r.embedding is not None else np.zeros(0, dtype=np.float32)

            rows.append({
                "person_name": r.person_name,
                "image_type": r.image_type,
                "embedding": ",".join(map(str, emb)),
                "detection_score": r.detection_score,
            })

            embeddings.append(emb)
            labels.append(r.person_name)
            image_paths.append(r.image_path)

            parquet_rows["person_name"].append(r.person_name)
            parquet_rows["image_type"].append(r.image_type)
            parquet_rows["embedding"].append(emb.tolist())
            parquet_rows["detection_score"].append(r.detection_score)
            parquet_rows["quality_score"].append(r.quality_score)
            parquet_rows["embedding_norm"].append(r.embedding_norm)
            parquet_rows["image_path"].append(r.image_path)

        # Save to temp files
        csv_path = output_dir / "face_embeddings.csv"
        npz_path = output_dir / "face_embeddings.npz"
        parquet_path = output_dir / "face_embeddings.parquet"

        StorageAdapter.save_embeddings_csv(csv_path, rows)
        StorageAdapter.save_embeddings_npz(
            npz_path,
            np.array(embeddings, dtype=np.float32),
            np.array(labels),
            np.array(image_paths)
        )

        parquet_written = None
        if enable_parquet:
            if StorageAdapter.save_parquet(parquet_path, parquet_rows):
                parquet_written = parquet_path

        return {"csv": csv_path, "npz": npz_path, "parquet": parquet_written}

    # ==================== SPLITS API (Temp Processing) ====================
    # Save training splits to temp files

    @staticmethod
    def save_split_indices(path: Path, train_idx: np.ndarray, val_idx: np.ndarray, 
                          test_idx: np.ndarray) -> bool:
        """Save train/val/test split indices to temp file."""
        try:
            StorageAdapter.ensure_dir(path.parent)
            np.savez_compressed(path, train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)
            logger.debug(f"Saved split indices: {path}")
            return True
        except Exception as ex:
            logger.error(f"save_split_indices failed: {ex}")
            return False

    @staticmethod
    def save_split_csv(path: Path, df: pd.DataFrame) -> bool:
        """Save split to temp CSV."""
        try:
            StorageAdapter.ensure_dir(path.parent)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            logger.debug(f"Saved split CSV: {path} rows={len(df)}")
            return True
        except Exception as ex:
            logger.error(f"save_split_csv failed: {ex}")
            return False

    @staticmethod
    def save_split_npz(path: Path, embeddings: np.ndarray, labels: np.ndarray, 
                      image_paths: np.ndarray) -> bool:
        """Save split to temp NPZ."""
        try:
            StorageAdapter.ensure_dir(path.parent)
            np.savez_compressed(path, embeddings=embeddings, labels=labels, image_paths=image_paths)
            logger.debug(f"Saved split NPZ: {path} shape={embeddings.shape}")
            return True
        except Exception as ex:
            logger.error(f"save_split_npz failed: {ex}")
            return False

    @staticmethod
    def save_split_bundle(splits_dir: Optional[Path], split_name: str, embeddings: np.ndarray,
                         labels: np.ndarray, image_paths: np.ndarray, df: pd.DataFrame) -> Dict[str, Path]:
        """Save train/val/test split bundle to temp files."""
        splits_dir = splits_dir or (TEMP_DIR / "splits")  # ✅ Use TEMP_DIR
        StorageAdapter.ensure_dir(splits_dir)

        npz_path = splits_dir / f"{split_name}.npz"
        csv_path = splits_dir / f"{split_name}.csv"

        StorageAdapter.save_split_npz(npz_path, embeddings, labels, image_paths)
        StorageAdapter.save_split_csv(csv_path, df)

        return {"npz": npz_path, "csv": csv_path}

    # ==================== UTILITY ====================

    @staticmethod
    def save_json(path: Path, data: Dict) -> bool:
        """Save JSON to temp file."""
        try:
            StorageAdapter.ensure_dir(path.parent)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            logger.debug(f"Saved JSON: {path}")
            return True
        except Exception as ex:
            logger.error(f"save_json failed: {ex}")
            return False

    @staticmethod
    def file_size_mb(path: Path) -> float:
        """Get file size in MB."""
        return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0

    @staticmethod
    def path_exists(path: Path) -> bool:
        """Check if local path exists."""
        return path.exists()


__all__ = ["StorageAdapter"]

