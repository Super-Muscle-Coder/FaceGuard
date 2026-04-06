"""Realtime packaging service (Phase 6) with PySide6 GUI."""

from __future__ import annotations

import io
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from config.settings import (
    PACKAGER_CONFIG,
    PACKAGER_RUNTIME_KEYS,
    EMBEDDING_PATHS,
    DATABASE_DIR,
)
from core.adapters.ModelAdapter import ModelAdapter
from core.adapters.StorageAdapter import StorageAdapter
from core.entities import DetectedFace, FrameStatistics
from core.storage import get_sqlite_manager

logger = logging.getLogger(__name__)


class _FaceTracker:
    def __init__(self, iou_threshold: float, smoothing_factor: float):
        self.iou_threshold = iou_threshold
        self.smoothing_factor = smoothing_factor
        self.previous: List[DetectedFace] = []

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / max(1.0, area_a + area_b - inter)

    def _smooth(self, old_bbox: Tuple[int, int, int, int], new_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        a = self.smoothing_factor
        return tuple(int(a * o + (1 - a) * n) for o, n in zip(old_bbox, new_bbox))

    def update(self, faces: List[DetectedFace]) -> List[DetectedFace]:
        if not faces:
            return self.previous
        if not self.previous:
            for f in faces:
                f.smooth_bbox = f.bbox
            self.previous = faces
            return faces

        updated: List[DetectedFace] = []
        for face in faces:
            best: Optional[DetectedFace] = None
            best_iou = 0.0
            for prev in self.previous:
                val = self._iou(face.bbox, prev.display_bbox)
                if val > best_iou and val >= self.iou_threshold:
                    best_iou = val
                    best = prev
            if best is not None:
                face.smooth_bbox = self._smooth(best.display_bbox, face.bbox)
            else:
                face.smooth_bbox = face.bbox
            updated.append(face)

        self.previous = updated
        return updated


class PackagingService:
    """Service layer for realtime test runtime."""

    def __init__(self, storage: Optional[StorageAdapter] = None):
        self.config = PACKAGER_CONFIG
        self.storage = storage or StorageAdapter()
        self.model_adapter = ModelAdapter(
            scrfd_path=EMBEDDING_PATHS["SCRFD_MODEL"],
            arcface_path=EMBEDDING_PATHS["ARCFACE_MODEL"],
        )
        self.tracker = _FaceTracker(
            iou_threshold=self.config["TRACKER_IOU_THRESHOLD"],
            smoothing_factor=self.config["TRACKER_SMOOTHING_FACTOR"],
        )
        self.threshold = float(self.config["DEFAULT_THRESHOLD"])
        self.database = self._load_database()
        self.class_to_index: Dict[str, int] = {}
        self._idx_to_class: Dict[int, str] = {}
        self.ft_head: Optional[torch.nn.Module] = None
        self._load_finetune_head()
        self.frame_index = 0

    # ==================== DATABASE ====================

    def _load_database(self) -> Dict[str, np.ndarray]:
        key = PACKAGER_RUNTIME_KEYS["DATABASE_NPZ"]

        # 1) MinIO
        # Always read runtime DB directly from MinIO to avoid stale local cache.
        raw = self.storage.get(key, use_cache=False)
        if raw:
            try:
                npz = np.load(io.BytesIO(raw), allow_pickle=False)
                db = {name: np.asarray(npz[name], dtype=np.float32) for name in npz.files}
                if db:
                    logger.info("Loaded realtime DB from MinIO: %d persons", len(db))
                    return db
            except Exception as ex:
                logger.warning("Failed to parse MinIO realtime DB: %s", ex)

        # 1.1) If MinIO has no runtime DB, do not silently use stale local unless explicitly allowed
        if not self.config.get("ALLOW_LOCAL_DB_FALLBACK", False):
            logger.warning("Realtime DB not found in MinIO and local fallback is disabled.")
            return {}

        # 2) Local fallback
        local_npz = DATABASE_DIR / "face_recognition_db.npz"
        if local_npz.exists():
            npz = np.load(local_npz, allow_pickle=False)
            db = {name: np.asarray(npz[name], dtype=np.float32) for name in npz.files}
            if db:
                logger.info("Loaded realtime DB from local file: %d persons", len(db))
                return db

        logger.warning("Realtime DB not found (MinIO/local).")
        return {}

    def reload_database(self):
        self.database = self._load_database()
        self._load_finetune_head()

    def set_threshold(self, threshold: float):
        self.threshold = float(threshold)

    # ==================== RECOGNITION ====================

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _recognize_embedding(self, embedding: np.ndarray) -> Tuple[str, float]:
        if not self.database:
            return "Unknown", 0.0
        best_name = "Unknown"
        best_score = -1.0
        for name, center in self.database.items():
            score = self._cosine(embedding, center)
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= self.threshold:
            return best_name, best_score
        return "Unknown", best_score

    def _load_finetune_head(self):
        self.ft_head = None
        self.class_to_index = {}
        self._idx_to_class = {}

        if not self.config.get("USE_FINETUNE_HEAD", True):
            return

        ckpt_path = DATABASE_DIR / "fine_tune_head.pt"
        if not ckpt_path.exists():
            logger.info("Fine-tune head checkpoint not found: %s", ckpt_path)
            return

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            class_to_index = ckpt.get("class_to_index") or {}
            state_dict = ckpt.get("state_dict") or {}
            if not class_to_index or "weight" not in state_dict:
                logger.warning("Invalid fine-tune head checkpoint format: %s", ckpt_path)
                return

            num_classes = len(class_to_index)
            input_dim = int(state_dict["weight"].shape[1])
            head = torch.nn.Linear(input_dim, num_classes)
            head.load_state_dict(state_dict)
            head.eval()

            self.ft_head = head
            self.class_to_index = {str(k): int(v) for k, v in class_to_index.items()}
            self._idx_to_class = {v: k for k, v in self.class_to_index.items()}
            logger.info("Loaded fine-tune head checkpoint: %s (classes=%d)", ckpt_path, num_classes)
        except Exception as ex:
            logger.warning("Failed to load fine-tune head checkpoint: %s", ex)

    def _head_predict(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
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

    def _recognize_hybrid(self, embedding: np.ndarray) -> Tuple[str, float]:
        cos_name, cos_conf = self._recognize_embedding(embedding)
        head_name, head_conf = self._head_predict(embedding)

        # Safety: if runtime cosine DB is empty, never allow head-only recognition
        # to avoid recognizing identities that were removed from MinIO/SQLite.
        if not self.database:
            return "Unknown", 0.0

        if head_name is None:
            return cos_name, cos_conf

        if cos_name == "Unknown":
            if head_conf >= self.config.get("FINETUNE_HEAD_OVERRIDE_CONF", 0.80):
                return head_name, head_conf
            return cos_name, cos_conf

        # If both agree, blend confidence
        if head_name == cos_name:
            a = float(self.config.get("FINETUNE_HEAD_BLEND_ALPHA", 0.35))
            merged = (1.0 - a) * float(cos_conf) + a * float(head_conf)
            return cos_name, float(merged)

        # If disagree, allow override only when head is very confident and cosine is weak
        override_thr = float(self.config.get("FINETUNE_HEAD_OVERRIDE_CONF", 0.80))
        if head_conf >= override_thr and cos_conf < max(self.threshold, 0.65):
            return head_name, float(head_conf)

        return cos_name, cos_conf

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[List[DetectedFace], FrameStatistics]:
        start = time.perf_counter()
        detected = self.model_adapter.detect_faces(frame_bgr, threshold=0.5, return_dataclass=False)
           
        faces: List[DetectedFace] = []
        for item in detected[: self.config["MAX_FACES_PER_FRAME"]]:
            bbox = tuple(int(v) for v in item["bbox"])
            d = DetectedFace(bbox=bbox, confidence=float(item.get("score", 0.0)))

            aligned = self.model_adapter.align_face(frame_bgr, item.get("landmarks"))
            if aligned is not None:
                embedding = self.model_adapter.extract_embedding(aligned)
            else:
                x1, y1, x2, y2 = bbox
                crop = frame_bgr[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                embedding = self.model_adapter.extract_embedding(cv2.resize(crop, (112, 112))) if crop.size else None

            if embedding is not None:
                d.embedding = embedding.astype(np.float32)
                identity, conf = self._recognize_hybrid(d.embedding)

                # Final sync guard with SQLite active persons
                if identity != "Unknown":
                    try:
                        p = get_sqlite_manager().get_person_by_name(identity)
                        if not p or p.get("status") != "active":
                            identity, conf = "Unknown", 0.0
                    except Exception:
                        identity, conf = "Unknown", 0.0

                d.identity = identity
                d.recognition_confidence = float(max(conf, 0.0))

                try:
                    sqlite = get_sqlite_manager()
                    person_id = None
                    if identity != "Unknown":
                        person = sqlite.get_person_by_name(identity)
                        person_id = person["person_id"] if person else None
                    sqlite.add_access_log(
                        person_id=person_id,
                        camera_id=f"cam_{self.config['CAMERA_ID']}",
                        result=identity,
                        confidence=d.recognition_confidence,
                        metadata={"bbox": list(d.bbox)},
                    )
                except Exception:
                    pass

            faces.append(d)

        faces = self.tracker.update(faces)
        self.frame_index += 1

        known = sum(1 for f in faces if f.is_known)
        unknown = len(faces) - known
        confs = [f.recognition_confidence for f in faces if f.recognition_confidence is not None]

        stats = FrameStatistics(
            frame_count=self.frame_index,
            faces_detected=len(faces),
            known_faces=known,
            unknown_faces=unknown,
            avg_confidence=float(np.mean(confs)) if confs else 0.0,
            processing_time=time.perf_counter() - start,
        )
        return faces, stats


# ==================== GUI (PySide6) ====================


def _to_qpixmap(frame_bgr):
    from PySide6.QtGui import QImage, QPixmap

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _draw_faces(frame: np.ndarray, faces: List[DetectedFace], cfg: Dict):
    known_color = tuple(cfg["KNOWN_COLOR"])
    unknown_color = tuple(cfg["UNKNOWN_COLOR"])

    for f in faces:
        x1, y1, x2, y2 = f.display_bbox
        color = known_color if f.is_known else unknown_color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, int(cfg["BOX_THICKNESS"]))
        text = "Detecting..."
        if f.identity is not None:
            if f.is_known:
                text = f"{f.identity} ({(f.recognition_confidence or 0.0) * 100:.1f}%)"
            else:
                text = f"Unknown ({(f.recognition_confidence or 0.0) * 100:.1f}%)"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(cfg["FONT_SCALE"]),
            color,
            2,
            cv2.LINE_AA,
        )


class _PackagerWindow:
    def __init__(self, service: PackagingService):
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
            QFrame,
        )

        self.service = service
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_tick = time.perf_counter()

        self.window = QMainWindow()
        self.window.setWindowTitle("FaceGuard Realtime Packager")
        self.window.resize(
            int(service.config["WINDOW_WIDTH"]),
            int(service.config["WINDOW_HEIGHT"]),
        )

        central = QWidget()
        self.window.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.video_label = QLabel("Camera not started")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(860, 640)
        self.video_label.setStyleSheet("background:#0f172a;border:1px solid #334155;border-radius:12px;color:#cbd5e1;")

        panel = QFrame()
        panel.setFixedWidth(320)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(12)

        title = QLabel("FaceGuard Runtime")
        title.setObjectName("title")

        self.threshold_label = QLabel(f"Threshold: {self.service.threshold:.2f}")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(40)
        self.threshold_slider.setMaximum(90)
        self.threshold_slider.setValue(int(self.service.threshold * 100))
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)

        self.btn_start = QPushButton("Start Camera")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_reload = QPushButton("Reload DB")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_reload.clicked.connect(self.reload_db)

        self.stats_label = QLabel("Stats: -")
        self.db_label = QLabel(f"DB Persons: {len(self.service.database)}")

        panel_layout.addWidget(title)
        panel_layout.addWidget(self.threshold_label)
        panel_layout.addWidget(self.threshold_slider)
        panel_layout.addWidget(self.btn_start)
        panel_layout.addWidget(self.btn_stop)
        panel_layout.addWidget(self.btn_reload)
        panel_layout.addWidget(self.db_label)
        panel_layout.addWidget(self.stats_label)
        panel_layout.addStretch(1)

        layout.addWidget(self.video_label, 1)
        layout.addWidget(panel)

        self.window.setStyleSheet(
            """
            QMainWindow, QWidget { background:#020617; color:#e2e8f0; font-size:13px; }
            QLabel#title { font-size:20px; font-weight:700; color:#f8fafc; }
            QPushButton { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:10px; }
            QPushButton:hover { background:#334155; }
            QPushButton:disabled { background:#0f172a; color:#64748b; }
            QSlider::groove:horizontal { height:8px; background:#1e293b; border-radius:4px; }
            QSlider::handle:horizontal { width:16px; background:#38bdf8; margin:-4px 0; border-radius:8px; }
            """
        )

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(30)

    def show(self):
        self.window.show()

    def _on_threshold_changed(self, value: int):
        threshold = value / 100.0
        self.service.set_threshold(threshold)
        self.threshold_label.setText(f"Threshold: {threshold:.2f}")

    def reload_db(self):
        self.service.reload_database()
        self.db_label.setText(f"DB Persons: {len(self.service.database)}")

    def start_camera(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(int(self.service.config["CAMERA_ID"]))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.service.config["CAMERA_WIDTH"]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.service.config["CAMERA_HEIGHT"]))

        if not self.cap.isOpened():
            self.video_label.setText("Cannot open camera")
            self.cap = None
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start()

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _tick(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            return

        faces, stats = self.service.process_frame(frame)

        now = time.perf_counter()
        dt = max(1e-6, now - self.last_tick)
        self.last_tick = now
        stats.fps = 1.0 / dt

        _draw_faces(frame, faces, self.service.config)
        pix = _to_qpixmap(frame)
        self.video_label.setPixmap(pix)

        self.stats_label.setText(
            f"FPS: {stats.fps:.1f}\n"
            f"Faces: {stats.faces_detected}\n"
            f"Known/Unknown: {stats.known_faces}/{stats.unknown_faces}\n"
            f"Avg Conf: {stats.avg_confidence * 100:.1f}%\n"
            f"Proc: {stats.processing_time * 1000:.1f} ms"
        )


def launch_packaging_gui(storage: Optional[StorageAdapter] = None) -> int:
    """Launch phase-6 realtime GUI."""
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as ex:
        logger.error("PySide6 chưa được cài hoặc lỗi import: %s", ex)
        return 1

    app = QApplication.instance() or QApplication(sys.argv)
    service = PackagingService(storage=storage)
    window = _PackagerWindow(service)
    window.show()
    return app.exec()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return launch_packaging_gui()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PackagingService", "launch_packaging_gui"]
