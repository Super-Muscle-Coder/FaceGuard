"""IoT service for ESP32CAM runtime recognition."""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from config.settings import EMBEDDING_PATHS, IOT_MINIO_KEYS, IOT_SERVICE_CONFIG, DATABASE_DIR
from core.adapters.ModelAdapter import ModelAdapter
from core.adapters.StorageAdapter import StorageAdapter
from core.storage import get_sqlite_manager
from core.entities import CameraStatus, IoTRecognitionResult, ServiceMetrics

logger = logging.getLogger(__name__)


class IoTService:
    def __init__(self, storage: Optional[StorageAdapter] = None):
        self.storage = storage or StorageAdapter()
        self.model_adapter = ModelAdapter(
            scrfd_path=EMBEDDING_PATHS["SCRFD_MODEL"],
            arcface_path=EMBEDDING_PATHS["ARCFACE_MODEL"],
        )

        self.threshold = float(IOT_SERVICE_CONFIG["DEFAULT_RECOGNITION_THRESHOLD"])
        self.database: Dict[str, np.ndarray] = {}
        self.class_to_index: Dict[str, int] = {}
        self._idx_to_class: Dict[int, str] = {}
        self.ft_head: Optional[torch.nn.Module] = None

        self.metrics = ServiceMetrics(started_at=time.time())
        self.camera_statuses: Dict[str, CameraStatus] = {}
        self.last_frames: Dict[str, bytes] = {}
        self.last_frame_ts: Dict[str, float] = {}

        self.reload_database(force_refresh=True)
        self._load_finetune_head()
        self._sqlite = get_sqlite_manager()

    def _get_active_person_names(self) -> set[str]:
        try:
            rows = self._sqlite.list_persons(status="active")
            return {str(r.get("name")) for r in rows if r.get("name")}
        except Exception:
            return set()

    def _get_valid_runtime_names(self) -> set[str]:
        """Intersection of runtime DB identities and active SQLite identities."""
        if not self.database:
            return set()
        active_names = self._get_active_person_names()
        if not active_names:
            return set(self.database.keys())
        return set(self.database.keys()) & active_names

    @staticmethod
    def _is_valid_identity(name: Optional[str], valid_names: set[str]) -> bool:
        return bool(name) and name in valid_names

    # ==================== DATABASE ====================

    def reload_database(self, force_refresh: bool = False):
        key = IOT_MINIO_KEYS["DATABASE_NPZ"]
        logger.info(
            "IoTService: reloading database key=%s force_refresh=%s use_cache=%s",
            key,
            force_refresh,
            not force_refresh,
        )
        raw = self.storage.get(key, use_cache=not force_refresh)
        if raw is None:
            logger.warning("IoTService: database npz not found in MinIO key=%s", key)
            self.database = {}
            return

        npz = np.load(io.BytesIO(raw), allow_pickle=False)
        loaded = {name: np.asarray(npz[name], dtype=np.float32) for name in npz.files}
        self.database = loaded
        logger.info("IoTService: loaded %d person embeddings", len(self.database))
        if self.database:
            sample = next(iter(self.database.values()))
            logger.info("IoTService: embedding_dim=%d", int(sample.shape[0]))

    def _load_finetune_head(self):
        self.ft_head = None
        self.class_to_index = {}
        self._idx_to_class = {}

        if not IOT_SERVICE_CONFIG.get("USE_FINETUNE_HEAD", True):
            return

        ckpt_path = DATABASE_DIR / "fine_tune_head.pt"
        if not ckpt_path.exists():
            logger.info("IoTService: fine-tune head checkpoint not found: %s", ckpt_path)
            return

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            class_to_index = ckpt.get("class_to_index") or {}
            state_dict = ckpt.get("state_dict") or {}
            if not class_to_index or "weight" not in state_dict:
                logger.warning("IoTService: invalid fine-tune checkpoint format: %s", ckpt_path)
                return

            num_classes = len(class_to_index)
            input_dim = int(state_dict["weight"].shape[1])
            head = torch.nn.Linear(input_dim, num_classes)
            head.load_state_dict(state_dict)
            head.eval()

            self.ft_head = head
            self.class_to_index = {str(k): int(v) for k, v in class_to_index.items()}
            self._idx_to_class = {v: k for k, v in self.class_to_index.items()}
            logger.info("IoTService: loaded fine-tune head checkpoint classes=%d", num_classes)
        except Exception as ex:
            logger.warning("IoTService: failed to load fine-tune head checkpoint: %s", ex)

    # ==================== METRICS/STATUS ====================

    def get_metrics(self) -> ServiceMetrics:
        return self.metrics

    def get_camera_statuses(self) -> Dict[str, CameraStatus]:
        return self.camera_statuses

    # ==================== RECOGNITION ====================

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _recognize_embedding(self, embedding: np.ndarray) -> tuple[str, float]:
        if not self.database:
            return "Unknown", 0.0

        valid_names = self._get_valid_runtime_names()
        if not valid_names:
            logger.warning("IoTService: no valid runtime identities (db=%d)", len(self.database))
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0
        for name, center in self.database.items():
            if name not in valid_names:
                continue
            score = self._cosine(embedding, center)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.threshold:
            return best_name, best_score
        return "Unknown", best_score

    def _head_predict(self, embedding: np.ndarray) -> tuple[Optional[str], float]:
        if self.ft_head is None:
            return None, 0.0
        try:
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.ft_head(x)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            idx_val = int(idx.item())
            return self._idx_to_class.get(idx_val), float(conf.item())
        except Exception:
            return None, 0.0

    def _recognize_hybrid(self, embedding: np.ndarray) -> tuple[str, float]:
        cos_name, cos_conf = self._recognize_embedding(embedding)
        head_name, head_conf = self._head_predict(embedding)

        if not self.database:
            return "Unknown", 0.0

        valid_names = self._get_valid_runtime_names()
        if not valid_names:
            return "Unknown", 0.0

        if head_name is None or not self._is_valid_identity(head_name, valid_names):
            return cos_name, cos_conf

        if cos_name == "Unknown":
            if head_conf >= float(IOT_SERVICE_CONFIG.get("FINETUNE_HEAD_OVERRIDE_CONF", 0.80)):
                return head_name, head_conf
            return cos_name, cos_conf

        if head_name == cos_name and self._is_valid_identity(cos_name, valid_names):
            a = float(IOT_SERVICE_CONFIG.get("FINETUNE_HEAD_BLEND_ALPHA", 0.35))
            merged = (1.0 - a) * float(cos_conf) + a * float(head_conf)
            return cos_name, float(merged)

        override_thr = float(IOT_SERVICE_CONFIG.get("FINETUNE_HEAD_OVERRIDE_CONF", 0.80))
        if head_conf >= override_thr and cos_conf < max(self.threshold, 0.65):
            return head_name, float(head_conf)

        return (cos_name, cos_conf) if self._is_valid_identity(cos_name, valid_names) else ("Unknown", 0.0)

    def recognize_image_bytes(self, image_bytes: bytes, camera_id: str = "unknown") -> IoTRecognitionResult:
        start = time.perf_counter()
        logger.debug("IoTService: recognize request camera_id=%s payload_bytes=%d", camera_id, len(image_bytes))

        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Invalid image bytes")

            # Keep latest frame for debug stream endpoint.
            self.last_frames[camera_id] = image_bytes
            self.last_frame_ts[camera_id] = time.time()

            faces = self.model_adapter.detect_faces(frame, threshold=0.5, return_dataclass=False)
            if not faces:
                logger.info("IoTService: no face detected camera_id=%s", camera_id)
                result = IoTRecognitionResult(
                    status="denied",
                    identity="Unknown",
                    confidence=0.0,
                    message="No face detected",
                )
                self.metrics.failed_requests += 1
                return result

            # largest face
            face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
            aligned = self.model_adapter.align_face(frame, face.get("landmarks"))
            if aligned is None:
                x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                m = float(IOT_SERVICE_CONFIG.get("FACE_CROP_MARGIN_RATIO", 0.15))
                mx = int(w * m)
                my = int(h * m)
                x1 = max(0, x1 - mx)
                y1 = max(0, y1 - my)
                x2 = min(frame.shape[1], x2 + mx)
                y2 = min(frame.shape[0], y2 + my)
                crop = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                if crop.size == 0:
                    raise ValueError("Invalid face crop")
                aligned = cv2.resize(crop, (112, 112))

            if bool(IOT_SERVICE_CONFIG.get("ENABLE_CLAHE_ENHANCE", True)):
                ycrcb = cv2.cvtColor(aligned, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                y = clahe.apply(y)
                aligned = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

            embedding = self.model_adapter.extract_embedding(aligned)
            if embedding is None:
                raise ValueError("Embedding extraction failed")

            identity, confidence = self._recognize_hybrid(np.asarray(embedding, dtype=np.float32))

            # If identity unknown but similarity crossed threshold, keep best score for diagnostics
            if identity == "Unknown" and confidence >= self.threshold:
                logger.warning(
                    "IoTService: score>=threshold but identity unknown after sync guard camera_id=%s score=%.4f threshold=%.2f",
                    camera_id,
                    confidence,
                    self.threshold,
                )

            if identity != "Unknown":
                try:
                    p = get_sqlite_manager().get_person_by_name(identity)
                    if not p or p.get("status") != "active":
                        identity, confidence = "Unknown", 0.0
                except Exception:
                    identity, confidence = "Unknown", 0.0
            allowed = identity != "Unknown" and confidence >= self.threshold

            result = IoTRecognitionResult(
                status="allowed" if allowed else "denied",
                identity=identity,
                confidence=float(max(confidence, 0.0)),
                message="Matched" if allowed else "Not matched",
            )

            if allowed:
                self.metrics.success_requests += 1
                logger.info(
                    "IoTService: matched camera_id=%s identity=%s confidence=%.4f threshold=%.2f",
                    camera_id,
                    identity,
                    confidence,
                    self.threshold,
                )
            else:
                self.metrics.failed_requests += 1
                logger.info(
                    "IoTService: not matched camera_id=%s best_identity=%s confidence=%.4f threshold=%.2f",
                    camera_id,
                    identity,
                    confidence,
                    self.threshold,
                )

            return result

        except Exception as ex:
            self.metrics.failed_requests += 1
            logger.exception("IoTService: recognition failed camera_id=%s error=%s", camera_id, ex)
            return IoTRecognitionResult(
                status="denied",
                identity="Unknown",
                confidence=0.0,
                message=str(ex),
            )

        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.metrics.update_latency(latency_ms)
            logger.debug("IoTService: request done camera_id=%s latency_ms=%.2f", camera_id, latency_ms)

            status = self.camera_statuses.get(camera_id)
            if status is None:
                status = CameraStatus(camera_id=camera_id)
                self.camera_statuses[camera_id] = status
            status.last_seen = datetime.now().isoformat()
            status.requests += 1

    def process_request(self, image_bytes: bytes, camera_id: str = "unknown") -> Dict:
        result = self.recognize_image_bytes(image_bytes=image_bytes, camera_id=camera_id)

        status = self.camera_statuses.get(camera_id)
        if status is not None:
            status.last_identity = result.identity
            status.last_confidence = float(result.confidence)
            status.last_result = result.status

        payload = result.to_dict()
        payload["camera_id"] = camera_id
        payload["database_persons"] = len(self.database)
        payload["threshold"] = float(self.threshold)
        return payload

    def get_debug_runtime_state(self) -> Dict:
        valid_names = sorted(self._get_valid_runtime_names())
        active_names = sorted(self._get_active_person_names())
        db_names = sorted(self.database.keys())
        return {
            "threshold": float(self.threshold),
            "database_persons": len(self.database),
            "database_names": db_names,
            "active_sqlite_names": active_names,
            "valid_runtime_names": valid_names,
            "finetune_head_loaded": self.ft_head is not None,
            "head_classes": sorted(self.class_to_index.keys()),
        }

    def get_last_frame_jpeg(self, camera_id: str) -> bytes | None:
        raw = self.last_frames.get(camera_id)
        ts = self.last_frame_ts.get(camera_id, 0.0)
        if raw is None:
            return None
        if (time.time() - ts) > float(IOT_SERVICE_CONFIG.get("STREAM_STALE_SECONDS", 10)):
            return None

        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        q = int(IOT_SERVICE_CONFIG.get("STREAM_JPEG_QUALITY", 80))
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return None
        return enc.tobytes()


__all__ = ["IoTService"]