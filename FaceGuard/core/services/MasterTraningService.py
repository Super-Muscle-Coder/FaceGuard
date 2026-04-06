"""Master training service for fine-tune research pipeline.

Pipeline (research):
- Phase 1: Data Collection (GUI)
- Phase 2: Frame Extraction
- Phase 3: Frame Sanitization
- Phase 4: Fine-tune ArcFace head + runtime sync
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from config.settings import (
    EMBEDDING_PATHS,
    TEMP_DIR,
    get_frame_type_dir,
)
from core.adapters.ModelAdapter import ModelAdapter
from core.adapters.StorageAdapter import StorageAdapter
from core.adapters.VideoAdapter import VideoAdapter
from core.services.DataCollectionService import DataCollectionService
from core.services.FrameExtractionService import FrameExtractionService
from core.services.FrameSanitizerService import FrameSanitizerService
from core.services.FineTuneService import FineTuneService

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


class MasterTraningService:
    def __init__(
        self,
        video_adapter: Optional[VideoAdapter] = None,
        model_adapter: Optional[ModelAdapter] = None,
        storage_adapter: Optional[StorageAdapter] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.video_adapter = video_adapter or VideoAdapter()
        self.model_adapter = model_adapter or self._init_model_adapter()
        self.storage = storage_adapter or StorageAdapter()
        self.progress_callback = progress_callback

        self.person_name: Optional[str] = None
        self.video_paths: Optional[dict[str, str]] = None
        self._fine_tune_service: Optional[FineTuneService] = None
        self.last_finetune_report = None

    def _log_progress(self, phase: int, message: str):
        logger.info("[Phase %d] %s", phase, message)
        if self.progress_callback:
            try:
                self.progress_callback(phase, message)
            except Exception:
                pass

    def _init_model_adapter(self) -> ModelAdapter:
        scrfd_model = EMBEDDING_PATHS["SCRFD_MODEL"]
        arcface_model = EMBEDDING_PATHS["ARCFACE_MODEL"]

        if not scrfd_model.exists() or not arcface_model.exists():
            raise FileNotFoundError(
                f"Model files not found. SCRFD={scrfd_model}, ArcFace={arcface_model}"
            )

        return ModelAdapter(scrfd_path=scrfd_model, arcface_path=arcface_model)

    def run_phase1_data_collection(self, person_name: str, auto_reject_threshold: float = 0.6) -> bool:
        self.person_name = person_name
        self._log_progress(1, f"Starting data collection for: {person_name}")

        service = DataCollectionService(
            video_adapter=self.video_adapter,
            auto_reject_threshold=auto_reject_threshold,
        )
        self.video_paths = service.collect_videos(person_name)
        return bool(self.video_paths)

    def run_phase2_frame_extraction(self, target_frames: int = 100) -> bool:
        if not self.person_name or not self.video_paths:
            raise PipelineError("Phase 1 must be completed first")

        self._log_progress(2, "Extracting frames from videos...")
        service = FrameExtractionService(storage=self.storage)

        total_frames = 0
        for video_type, video_path in self.video_paths.items():
            output_dir = get_frame_type_dir(self.person_name, video_type)
            frame_paths, _ = service.extract_from_video(
                video_path=Path(video_path),
                person_name=self.person_name,
                video_type=video_type,
                output_dir=output_dir,
                target_frames=target_frames,
            )
            total_frames += len(frame_paths)

        self._log_progress(2, f"Extracted {total_frames} frames")
        return total_frames > 0

    def run_phase3_frame_sanitizer(self) -> bool:
        if not self.person_name:
            raise PipelineError("Phase 1 must be completed first")

        self._log_progress(3, "Sanitizing extracted frames...")
        service = FrameSanitizerService(
            model_adapter=self.model_adapter,
            person_name=self.person_name,
        )
        report = service.run()
        self._log_progress(3, f"Sanitized {report.total_output}/{report.total_input} frames")
        return report.total_output > 0

    def run_phase4_fine_tune(self) -> bool:
        if not self.person_name:
            raise PipelineError("Phase 1 must be completed first")

        self._log_progress(4, "Running fine-tune training...")
        self._fine_tune_service = FineTuneService(storage=self.storage, person_name=self.person_name)
        report = self._fine_tune_service.run_training_only()
        self.last_finetune_report = report
        self._log_progress(4, f"Fine-tune complete: best_val_acc={report.best_val_acc:.4f}")
        return True

    def has_reusable_sanitized_data(self, person_name: str) -> bool:
        base = TEMP_DIR / person_name / "sanitized_frames"
        if not base.exists():
            return False
        for angle in ("frontal", "horizontal", "vertical"):
            angle_dir = base / angle
            if angle_dir.exists() and any(angle_dir.glob("*.jpg")):
                return True
        return False

    def run_finetune_only_from_existing(self, person_name: str) -> bool:
        self.person_name = person_name
        if not self.has_reusable_sanitized_data(person_name):
            logger.error("No reusable sanitized data found for person=%s", person_name)
            return False
        return self.run_phase4_fine_tune()

    def deploy_last_finetune(self, cleanup_temp: bool = True) -> bool:
        if self._fine_tune_service is None or self.last_finetune_report is None:
            raise PipelineError("No fine-tune result available for deployment")
        self._fine_tune_service.deploy_after_training(
            self.last_finetune_report.class_to_index,
            cleanup_temp=cleanup_temp,
        )
        return True

    def run_complete_pipeline(
        self,
        person_name: str,
        auto_reject_threshold: float = 0.6,
        target_frames: int = 100,
    ) -> bool:
        try:
            if not self.run_phase1_data_collection(person_name, auto_reject_threshold):
                logger.error("Pipeline aborted: Phase 1 failed")
                return False

            if not self.run_phase2_frame_extraction(target_frames):
                logger.error("Pipeline aborted: Phase 2 failed")
                return False

            if not self.run_phase3_frame_sanitizer():
                logger.error("Pipeline aborted: Phase 3 failed")
                return False

            if not self.run_phase4_fine_tune():
                logger.error("Pipeline aborted: Phase 4 failed")
                return False

            logger.info("Fine-tune research pipeline completed successfully")
            return True

        except Exception as ex:
            logger.exception("Fine-tune research pipeline failed: %s", ex)
            return False


__all__ = ["MasterTraningService", "PipelineError"]

