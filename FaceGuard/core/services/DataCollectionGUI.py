"""Data Collection GUI (Phase 1) using PySide6.

Workflow:
1) Record 3 videos: frontal -> horizontal -> vertical
2) Each video is quality-checked immediately via VideoQualityService
3) User can Keep or Re-record each step
4) Returns dict[video_type, path] when all 3 are accepted
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2

from config.settings import DATA_COLLECTION_CONFIG, get_video_dir
from core.adapters import VideoAdapter
from core.services.VideoQualityService import VideoQualityService

logger = logging.getLogger(__name__)


_VIDEO_TYPES = ["frontal", "horizontal", "vertical"]
_VIDEO_HINTS = {
    "frontal": "Nhìn thẳng vào camera, giữ đầu ổn định.",
    "horizontal": "Quay đầu trái -> phải chậm và đều.",
    "vertical": "Ngẩng lên -> cúi xuống chậm và đều.",
}


class _DataCollectionDialog:
    def __init__(
        self,
        person_name: str,
        video_adapter: VideoAdapter,
        quality_service: Optional[VideoQualityService],
        auto_reject_threshold: float,
    ):
        from PySide6.QtCore import QTimer, Signal
        from PySide6.QtWidgets import (
            QDialog,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QVBoxLayout,
            QFrame,
            QTextEdit,
            QProgressBar,
        )

        class _Emitter(QDialog):
            record_finished = Signal(bool, float, tuple, float, str)

        self.person_name = person_name
        self.video_adapter = video_adapter
        self.quality_service = quality_service or VideoQualityService(storage=None)
        self.auto_reject_threshold = auto_reject_threshold

        self.current_index = 0
        self.pending_video_path: Optional[Path] = None
        self.result: Optional[Dict[str, str]] = None

        self._recording_thread: Optional[threading.Thread] = None
        self._is_recording = False
        self._is_playback = False
        self._playback_cap = None
        self._last_frame = None
        self._frame_lock = threading.Lock()

        self.dialog = QDialog()
        self.dialog.setWindowTitle(f"FaceGuard - Data Collection: {person_name}")
        self.dialog.resize(1240, 760)

        root = QHBoxLayout(self.dialog)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Video panel
        self.video_label = QLabel("Camera Preview")
        from PySide6.QtCore import Qt
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(860, 640)
        self.video_label.setStyleSheet(
            "background:#0f172a;border:1px solid #334155;border-radius:12px;color:#cbd5e1;"
        )
        root.addWidget(self.video_label, 1)

        # Side panel
        side = QFrame()
        side.setFixedWidth(330)
        side_layout = QVBoxLayout(side)
        side_layout.setSpacing(10)

        self.title_label = QLabel("Phase 1 - Data Collection")
        self.title_label.setObjectName("title")

        self.step_label = QLabel()
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)

        self.progress = QProgressBar()
        self.progress.setRange(0, len(_VIDEO_TYPES))
        self.progress.setValue(0)

        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(260)

        self.btn_record = QPushButton("Record")
        self.btn_keep = QPushButton("Keep & Next")
        self.btn_rerecord = QPushButton("Re-record")
        self.btn_cancel = QPushButton("Cancel")

        self.btn_keep.setEnabled(False)
        self.btn_rerecord.setEnabled(False)

        self.btn_record.clicked.connect(self._record_current)
        self.btn_keep.clicked.connect(self._keep_current)
        self.btn_rerecord.clicked.connect(self._rerecord_current)
        self.btn_cancel.clicked.connect(self._cancel)

        side_layout.addWidget(self.title_label)
        side_layout.addWidget(self.step_label)
        side_layout.addWidget(self.hint_label)
        side_layout.addWidget(self.progress)
        side_layout.addWidget(self.status_box)
        side_layout.addWidget(self.btn_record)
        side_layout.addWidget(self.btn_keep)
        side_layout.addWidget(self.btn_rerecord)
        side_layout.addWidget(self.btn_cancel)
        side_layout.addStretch(1)

        root.addWidget(side)

        self.dialog.setStyleSheet(
            """
            QDialog, QWidget { background:#020617; color:#e2e8f0; font-size:13px; }
            QLabel#title { font-size:20px; font-weight:700; color:#f8fafc; }
            QPushButton { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:10px; }
            QPushButton:hover { background:#334155; }
            QPushButton:disabled { background:#0f172a; color:#64748b; }
            QProgressBar { border:1px solid #334155; border-radius:8px; text-align:center; }
            QProgressBar::chunk { background:#22c55e; border-radius:8px; }
            QTextEdit { background:#0f172a; border:1px solid #334155; border-radius:10px; }
            """
        )

        self._accepted_paths: Dict[str, str] = {}

        self._signal_emitter = _Emitter()
        self._signal_emitter.record_finished.connect(self._on_record_finished)

        self.timer = QTimer()
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._update_preview)

        self._refresh_step_ui()

    def _log(self, message: str):
        self.status_box.append(message)

    def _refresh_step_ui(self):
        if self.current_index >= len(_VIDEO_TYPES):
            return
        video_type = _VIDEO_TYPES[self.current_index]
        self.step_label.setText(
            f"Step {self.current_index + 1}/{len(_VIDEO_TYPES)}: {video_type.upper()}"
        )
        self.hint_label.setText(_VIDEO_HINTS.get(video_type, ""))
        self.progress.setValue(self.current_index)
        self._log(f"\n[STEP] {video_type.upper()} - sẵn sàng ghi hình")

    def _next_video_path(self, video_type: str) -> Path:
        video_dir = get_video_dir(self.person_name, video_type)
        counter = 1
        while True:
            path = video_dir / f"video_{counter:03d}.mp4"
            if not path.exists():
                return path
            counter += 1

    def _compute_quality_score(self, report) -> float:
        score = 1.0
        if getattr(report, "validation_issues", None):
            score -= 0.5
        if getattr(report, "has_critical_exposure", False):
            score -= 0.15
        if getattr(report, "has_critical_blur", False):
            score -= 0.20
        if getattr(report, "has_critical_noise", False):
            score -= 0.15
        return max(0.0, min(1.0, score))

    def _start_playback(self, video_path: Path):
        if self._playback_cap is not None:
            self._playback_cap.release()
            self._playback_cap = None
        self._playback_cap = cv2.VideoCapture(str(video_path))
        self._is_playback = self._playback_cap.isOpened()
        if self._is_playback:
            self._log("[PLAYBACK] Đang phát lại video vừa quay...")

    def _stop_playback(self):
        self._is_playback = False
        if self._playback_cap is not None:
            self._playback_cap.release()
            self._playback_cap = None

    def _record_current(self):
        video_type = _VIDEO_TYPES[self.current_index]
        output_path = self._next_video_path(video_type)

        self.btn_record.setEnabled(False)
        self.btn_keep.setEnabled(False)
        self.btn_rerecord.setEnabled(False)

        self._log(f"[RECORD] {video_type}: bắt đầu quay {DATA_COLLECTION_CONFIG['RECORDING_DURATION']} giây...")

        self._stop_playback()
        self._is_recording = True

        def _worker():
            duration_sec = int(DATA_COLLECTION_CONFIG["RECORDING_DURATION"])
            fps = int(DATA_COLLECTION_CONFIG["VIDEO_FPS"])
            codec = str(DATA_COLLECTION_CONFIG["VIDEO_CODEC"])

            cap = self.video_adapter.cap
            if cap is None or not self.video_adapter.is_open():
                self._signal_emitter.record_finished.emit(False, 0.0, (0, 0), 0.0, str(output_path))
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                width, height = DATA_COLLECTION_CONFIG["VIDEO_RESOLUTION"]

            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                self._signal_emitter.record_finished.emit(False, 0.0, (0, 0), 0.0, str(output_path))
                return

            start = time.time()
            frames = 0
            ok = True

            try:
                while True:
                    frame_result = self.video_adapter.read_frame()
                    if frame_result is None:
                        ok = False
                        break

                    ret, frame = frame_result
                    if not ret or frame is None:
                        ok = False
                        break

                    writer.write(frame)
                    frames += 1

                    with self._frame_lock:
                        self._last_frame = frame.copy()

                    elapsed = time.time() - start
                    if elapsed >= duration_sec:
                        break
            finally:
                writer.release()

            actual_duration = time.time() - start
            if frames == 0:
                ok = False

            self._signal_emitter.record_finished.emit(ok, actual_duration, (width, height), float(fps), str(output_path))

        self._recording_thread = threading.Thread(target=_worker, daemon=True)
        self._recording_thread.start()

    def _on_record_finished(self, ok: bool, actual_duration: float, resolution: tuple, fps: float, output_path_str: str):
        from PySide6.QtWidgets import QMessageBox

        self._is_recording = False

        output_path = Path(output_path_str)
        if not ok:
            self._log("[ERROR] Ghi hình thất bại")
            self.btn_record.setEnabled(True)
            return

        self.pending_video_path = output_path
        self._log(
            f"[SAVE] Đã lưu tạm: {output_path.name} | {actual_duration:.1f}s | {resolution[0]}x{resolution[1]} @ {fps:.1f}fps"
        )

        self._start_playback(output_path)

        report = self.quality_service.analyze_video(output_path)
        quality_score = self._compute_quality_score(report)
        self._log(f"[QUALITY] score={quality_score:.2f} | auto_reject<{self.auto_reject_threshold:.2f}")

        if report.warnings:
            for w in report.warnings:
                self._log(f"[WARNING] {w}")

        if report.validation_issues:
            for issue in report.validation_issues:
                self._log(f"[FAIL] {issue}")

        auto_reject = quality_score < self.auto_reject_threshold or bool(report.validation_issues)

        if auto_reject:
            self._log("[REJECT] Video không đạt chất lượng. Vui lòng quay lại.")
            QMessageBox.warning(
                self.dialog,
                "Video Quality Failed",
                "Video chưa đạt chất lượng Gate 1.\nVui lòng chọn Re-record.",
            )
            self.btn_record.setEnabled(False)
            self.btn_keep.setEnabled(False)
            self.btn_rerecord.setEnabled(True)
        else:
            self._log("[PASS] Video đạt chất lượng. Bạn có thể Keep hoặc Re-record.")
            self.btn_keep.setEnabled(True)
            self.btn_rerecord.setEnabled(True)
            self.btn_record.setEnabled(False)

    def _keep_current(self):
        if self.pending_video_path is None:
            return

        self._stop_playback()

        video_type = _VIDEO_TYPES[self.current_index]
        self._accepted_paths[video_type] = str(self.pending_video_path)
        self._log(f"[KEEP] {video_type}: {self.pending_video_path.name}")

        self.pending_video_path = None
        self.current_index += 1

        if self.current_index >= len(_VIDEO_TYPES):
            self.progress.setValue(len(_VIDEO_TYPES))
            self.result = dict(self._accepted_paths)
            self._log("[SUCCESS] Hoàn thành 3 lượt quay.")
            self.dialog.accept()
            return

        self.btn_record.setEnabled(True)
        self.btn_keep.setEnabled(False)
        self.btn_rerecord.setEnabled(False)
        self._refresh_step_ui()

    def _rerecord_current(self):
        self._stop_playback()

        if self.pending_video_path and self.pending_video_path.exists():
            try:
                os.remove(self.pending_video_path)
                self._log(f"[DELETE] Xóa video tạm: {self.pending_video_path.name}")
            except Exception as ex:
                self._log(f"[WARNING] Không thể xóa video tạm: {ex}")

        self.pending_video_path = None
        self.btn_record.setEnabled(True)
        self.btn_keep.setEnabled(False)
        self.btn_rerecord.setEnabled(False)
        self._log("[STEP] Sẵn sàng quay lại lượt hiện tại.")

    def _cancel(self):
        self._stop_playback()
        self._log("[CANCEL] Người dùng đã hủy.")
        self.dialog.reject()

    def _update_preview(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap

        frame = None

        if self._is_recording:
            with self._frame_lock:
                if self._last_frame is not None:
                    frame = self._last_frame.copy()
        elif self._is_playback and self._playback_cap is not None:
            ok, pb_frame = self._playback_cap.read()
            if ok and pb_frame is not None:
                frame = pb_frame
            else:
                self._playback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok2, pb_frame2 = self._playback_cap.read()
                if ok2 and pb_frame2 is not None:
                    frame = pb_frame2
        else:
            frame_result = self.video_adapter.read_frame()
            if frame_result is not None:
                ok, cam_frame = frame_result
                if ok and cam_frame is not None:
                    frame = cam_frame

        if frame is None:
            if self._last_frame is None:
                return
            frame = self._last_frame
        else:
            self._last_frame = frame.copy()

        if self.current_index < len(_VIDEO_TYPES):
            video_type = _VIDEO_TYPES[self.current_index]
            tag = "PLAYBACK" if self._is_playback else ("RECORDING" if self._is_recording else "LIVE")
            cv2.putText(
                frame,
                f"{video_type.upper()} | {self.person_name} | {tag}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def run(self) -> Optional[Dict[str, str]]:
        self.timer.start()
        code = self.dialog.exec()
        self.timer.stop()
        self._stop_playback()
        if code == 1:
            return self.result
        return None


def _find_existing_videos(person_name: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for video_type in _VIDEO_TYPES:
        video_dir = get_video_dir(person_name, video_type)
        candidates = sorted(video_dir.glob("*.mp4"))
        if candidates:
            result[video_type] = str(candidates[-1])
    return result


def launch_data_collection_gui(
    person_name: str,
    video_adapter=None,
    quality_service=None,
    auto_reject_threshold: float = 0.6,
) -> Optional[Dict[str, str]]:
    """Launch data collection GUI for Phase 1.

    Returns dict[video_type -> video_path] when all 3 rounds are accepted.
    """
    existing = _find_existing_videos(person_name)
    if len(existing) == 3:
        logger.info("Using existing recorded videos for %s", person_name)
        return existing

    if video_adapter is None:
        video_adapter = VideoAdapter(camera_id=DATA_COLLECTION_CONFIG["CAMERA_ID"])
        if not video_adapter.open_camera():
            logger.error("Cannot open camera for DataCollectionGUI")
            return None

    if not video_adapter.is_open():
        if not video_adapter.open_camera():
            logger.error("Cannot open camera for DataCollectionGUI")
            return None

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as ex:
        logger.error("PySide6 unavailable: %s", ex)
        logger.warning("Fallback: GUI disabled. Vui lòng cung cấp sẵn 3 video trong temp folder.")
        fallback = _find_existing_videos(person_name)
        return fallback if len(fallback) == 3 else None

    app = QApplication.instance() or QApplication([])
    dialog = _DataCollectionDialog(
        person_name=person_name,
        video_adapter=video_adapter,
        quality_service=quality_service,
        auto_reject_threshold=auto_reject_threshold,
    )
    return dialog.run()


__all__ = ["launch_data_collection_gui"]
