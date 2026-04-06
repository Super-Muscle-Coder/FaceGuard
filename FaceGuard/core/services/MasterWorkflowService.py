"""Master Workflow Service - Complete Face Recognition Pipeline.

Orchestrates all phases (1-5) into a single automated workflow.

===================  ARCHITECTURE  ====================
Phase 1: Data Collection (MANUAL - User records videos)
  → User inputs name
  → Records 3 videos (frontal, horizontal, vertical)
  → Videos saved to: data/temp/{person}/video/{type}/

Phase 2: Frame Extraction (AUTO)
  → Extracts quality frames from videos
  → Frames saved to: data/temp/{person}/frames/{type}/
  → Metadata CSVs exported

Phase 3: Embedding Generation (AUTO)
  → Extracts ArcFace embeddings from frames
  → NPZ + JSON saved to: data/temp/{person}/vector/

Phase 4: Data Sanitization (AUTO)
  → Validates, cleans, splits embeddings
  → Splits saved to: data/temp/{person}/vector/split/{train|val|test}/

Phase 5: Recognition Training (AUTO)
  → Trains SimpleFaceRecognizer
  → Uploads NPZ to MinIO
  → Saves metadata to SQLite
  → Cleanup: DELETE data/temp/{person}/

===================  USAGE  ====================
Simple usage:
    service = MasterWorkflowService()
    success = service.run_complete_pipeline("Nghi")

Advanced usage with callbacks:
    def progress_callback(phase, message):
        print(f"[Phase {phase}] {message}")
    
    service = MasterWorkflowService(
        progress_callback=progress_callback,
        auto_cleanup=True
    )
    success = service.run_complete_pipeline("Nghi", interactive=False)

===================  FEATURES  ====================
- Automatic pipeline execution (Phase 2-5)
- Progress tracking with callbacks
- Comprehensive error handling
- Automatic cleanup of temp files
- Interactive quality gate (Phase 5)
- Production-ready logging
"""

import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

from config.settings import (
    get_person_temp_dir,
    get_person_frame_dir,
    get_frame_type_dir,
    get_person_vector_dir,
    get_person_metadata_dir,
    get_person_split_dir,
)
from core.adapters.ModelAdapter import ModelAdapter
from core.adapters.StorageAdapter import StorageAdapter
from core.adapters.VideoAdapter import VideoAdapter
from core.services.DataCollectionService import DataCollectionService
from core.services.FrameExtractionService import FrameExtractionService
from core.services.EmbeddingService import EmbeddingService
from core.services.SanitizerService import SanitizerService
from core.services.RecognitionService import RecognitionService

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


class MasterWorkflowService:
    """
    Master orchestrator for complete face recognition pipeline.
    
    Coordinates all 5 phases from video recording to trained model deployment.
    
    Architecture:
    - Phase 1: Manual (DataCollectionService GUI)
    - Phase 2-5: Automatic (this service orchestrates)
    - Cleanup: Automatic temp file deletion after success
    
    Features:
    - Progress tracking
    - Error recovery
    - Automatic cleanup
    - Interactive/non-interactive modes
    """
    
    def __init__(
        self,
        video_adapter: Optional[VideoAdapter] = None,
        model_adapter: Optional[ModelAdapter] = None,
        storage_adapter: Optional[StorageAdapter] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        auto_cleanup: bool = True
    ):
        """
        Initialize MasterWorkflowService.
        
        Args:
            video_adapter: VideoAdapter instance (optional)
            model_adapter: ModelAdapter instance (optional)
            storage_adapter: StorageAdapter instance (optional)
            progress_callback: Callback function(phase_num, message)
            auto_cleanup: Auto-delete temp files after success?
        """
        logger.info("\n%s", "="*70)
        logger.info("  MASTER WORKFLOW SERVICE INITIALIZATION")
        logger.info("%s", "="*70)
        
        # Initialize adapters
        self.video_adapter = video_adapter or VideoAdapter()
        self.model_adapter = model_adapter or self._init_model_adapter()

        # Lazy-load storage (only needed for Phase 5)
        self._storage_adapter = storage_adapter
        self._storage = None
        
        # Configuration
        self.progress_callback = progress_callback
        self.auto_cleanup = auto_cleanup
        
        # Tracking
        self.person_name: Optional[str] = None
        self.video_paths: Optional[Dict[str, str]] = None
        self.current_phase = 0
        
        logger.info("   Adapters initialized")
        logger.info("   Auto cleanup: %s", auto_cleanup)
        logger.info("%s", "="*70)
    
    @property
    def storage(self) -> StorageAdapter:
        """Lazy-load storage adapter (only when needed)."""
        if self._storage is None:
            if self._storage_adapter:
                self._storage = self._storage_adapter
            else:
                try:
                    self._storage = StorageAdapter()
                    logger.info("  StorageAdapter initialized")
                except Exception as ex:
                    logger.warning("  MinIO not available, using local storage fallback")
                    logger.debug("    Error: %s", ex)
                    # Create local-only storage adapter
                    from core.adapters.StorageAdapter import StorageAdapter as SA
                    self._storage = SA()  # Will use local backend by default
        return self._storage

    def _init_model_adapter(self) -> ModelAdapter:
        """Initialize ModelAdapter with models."""
        from config.settings import EMBEDDING_PATHS
        
        scrfd_model = EMBEDDING_PATHS['SCRFD_MODEL']
        arcface_model = EMBEDDING_PATHS['ARCFACE_MODEL']
        
        if not scrfd_model.exists() or not arcface_model.exists():
            logger.error("Models not found!")
            logger.error("  SCRFD: %s", scrfd_model)
            logger.error("  ArcFace: %s", arcface_model)
            raise FileNotFoundError(
                "Models not found. Run: python scripts/download_models.py"
            )
        
        return ModelAdapter(
            scrfd_path=scrfd_model,
            arcface_path=arcface_model
        )
    
    def _log_progress(self, phase: int, message: str):
        """Log progress and call callback if provided."""
        logger.info("[Phase %d] %s", phase, message)
        
        if self.progress_callback:
            try:
                self.progress_callback(phase, message)
            except Exception as ex:
                logger.warning("Progress callback error: %s", ex)
    
    # ==================== PHASE 1: DATA COLLECTION ====================
    
    def run_phase1_data_collection(
        self,
        person_name: str,
        auto_reject_threshold: float = 0.6
    ) -> bool:
        """
        Phase 1: Data Collection (MANUAL - User interaction).
        
        Launches GUI for user to record 3 videos with quality checks.
        
        Args:
            person_name: Person name
            auto_reject_threshold: Auto-reject quality threshold
            
        Returns:
            True if successful (all 3 videos recorded)
        """
        self.current_phase = 1
        self.person_name = person_name
        
        logger.info("\n%s", "="*70)
        logger.info("  PHASE 1: DATA COLLECTION")
        logger.info("%s", "="*70)
        
        self._log_progress(1, f"Starting data collection for: {person_name}")
        
        try:
            service = DataCollectionService(
                video_adapter=self.video_adapter,
                auto_reject_threshold=auto_reject_threshold
            )
            
            self.video_paths = service.collect_videos(person_name)
            
            if not self.video_paths:
                logger.error("Data collection failed or cancelled")
                return False
            
            logger.info("\n✓ Phase 1 Complete:")
            for vtype, path in self.video_paths.items():
                logger.info("  %s: %s", vtype, path)
            
            self._log_progress(1, f"Collected {len(self.video_paths)}/3 videos")
            return True
            
        except Exception as ex:
            logger.exception("Phase 1 failed: %s", ex)
            self._log_progress(1, f"FAILED: {ex}")
            return False
    
    # ==================== PHASE 2: FRAME EXTRACTION ====================
    
    def run_phase2_frame_extraction(
        self,
        target_frames: int = 100
    ) -> bool:
        """
        Phase 2: Frame Extraction (AUTO).
        
        Extracts quality frames from recorded videos.
        
        Args:
            target_frames: Target frames per video
            
        Returns:
            True if successful
        """
        if not self.person_name or not self.video_paths:
            raise PipelineError("Phase 1 must be completed first")
        
        self.current_phase = 2
        
        logger.info("\n%s", "="*70)
        logger.info("  PHASE 2: FRAME EXTRACTION")
        logger.info("%s", "="*70)
        
        self._log_progress(2, "Extracting frames from videos...")
        
        try:
            service = FrameExtractionService(storage=self.storage)            
            total_frames = 0
            
            for video_type, video_path in self.video_paths.items():
                self._log_progress(2, f"Processing {video_type} video...")
                
                output_dir = get_frame_type_dir(self.person_name, video_type)
                
                frame_paths, metadata_path = service.extract_from_video(
                    video_path=Path(video_path),
                    person_name=self.person_name,
                    video_type=video_type,
                    output_dir=output_dir,
                    target_frames=target_frames
                )
                
                logger.info("  ✓ %s: %d frames extracted", video_type, len(frame_paths))
                total_frames += len(frame_paths)
            
            logger.info("\n✓ Phase 2 Complete: %d total frames", total_frames)
            self._log_progress(2, f"Extracted {total_frames} frames")
            return True
            
        except Exception as ex:
            logger.exception("Phase 2 failed: %s", ex)
            self._log_progress(2, f"FAILED: {ex}")
            raise PipelineError(f"Frame extraction failed: {ex}")
    
    # ==================== PHASE 3: EMBEDDING GENERATION ====================
    
    def run_phase3_embedding_generation(self) -> bool:
        """
        Phase 3: Embedding Generation (AUTO).
        
        Extracts ArcFace embeddings from frames.
        
        Returns:
            True if successful
        """
        if not self.person_name:
            raise PipelineError("Phase 1 must be completed first")
        
        self.current_phase = 3
        
        logger.info("\n%s", "="*70)
        logger.info("  PHASE 3: EMBEDDING GENERATION")
        logger.info("%s", "="*70)
        
        self._log_progress(3, "Generating embeddings...")
        
        try:
            # Prepare image directories
            frame_base_dir = get_person_frame_dir(self.person_name)
            
            image_dirs = {
                'frontal': frame_base_dir / 'frontal',
                'horizontal': frame_base_dir / 'horizontal',
                'vertical': frame_base_dir / 'vertical',
            }
            
            # Prepare metadata directories (same locations)
            metadata_dirs = image_dirs.copy()
            
            # Output directories
            vector_dir = get_person_vector_dir(self.person_name)
            metadata_dir = get_person_metadata_dir(self.person_name)
            
            service = EmbeddingService(
                model_adapter=self.model_adapter,
                storage=self.storage,
                person_name=self.person_name,
                image_dirs=image_dirs,
                metadata_dirs=metadata_dirs,
                vector_dir=vector_dir,
                metadata_output_dir=metadata_dir
            )
            
            results, saved_paths = service.run()
            
            if not results:
                raise PipelineError("No embeddings generated")
            
            logger.info("\n✓ Phase 3 Complete:")
            logger.info("  Embeddings: %d", len(results))
            logger.info("  NPZ: %s", saved_paths.get('npz'))
            logger.info("  JSON: %s", saved_paths.get('json'))
            
            self._log_progress(3, f"Generated {len(results)} embeddings")
            return True
            
        except Exception as ex:
            logger.exception("Phase 3 failed: %s", ex)
            self._log_progress(3, f"FAILED: {ex}")
            raise PipelineError(f"Embedding generation failed: {ex}")
    
    # ==================== PHASE 4: DATA SANITIZATION ====================
    
    def run_phase4_data_sanitization(self) -> bool:
        """
        Phase 4: Data Sanitization (AUTO).
        
        Validates, cleans, and splits embeddings.
        
        Returns:
            True if successful
        """
        if not self.person_name:
            raise PipelineError("Phase 1 must be completed first")
        
        self.current_phase = 4
        
        logger.info("\n%s", "="*70)
        logger.info("  PHASE 4: DATA SANITIZATION")
        logger.info("%s", "="*70)
        
        self._log_progress(4, "Sanitizing and splitting data...")
        
        try:
            # Input paths
            vector_dir = get_person_vector_dir(self.person_name)
            npz_path = vector_dir / "face_embeddings.npz"
            json_path = vector_dir.parent / "metadata" / "face_embeddings_metadata.json"
            
            if not npz_path.exists():
                raise PipelineError(f"NPZ not found: {npz_path}")
            if not json_path.exists():
                raise PipelineError(f"JSON not found: {json_path}")
            
            # Output directory
            output_dir = get_person_split_dir(self.person_name)
            
            service = SanitizerService(
                storage=self.storage,
                person_name=self.person_name
            )
            
            val_report, clean_report, split_report = service.run(
                npz_path=str(npz_path),
                json_path=str(json_path),
                output_dir=str(output_dir)
            )
            
            logger.info("\n✓ Phase 4 Complete:")
            logger.info("  Original: %d samples", clean_report.original_size)
            logger.info("  Cleaned: %d samples", clean_report.cleaned_size)
            logger.info("  Train: %d, Val: %d, Test: %d",
                       split_report.train_size,
                       split_report.val_size,
                       split_report.test_size)
            
            self._log_progress(4, f"Split: {split_report.train_size}/{split_report.val_size}/{split_report.test_size}")
            return True
            
        except Exception as ex:
            logger.exception("Phase 4 failed: %s", ex)
            self._log_progress(4, f"FAILED: {ex}")
            raise PipelineError(f"Data sanitization failed: {ex}")
    
    # ==================== PHASE 5: RECOGNITION TRAINING ====================
    
    def run_phase5_recognition_training(
        self,
        interactive: bool = True
    ) -> bool:
        """
        Phase 5: Recognition Training (AUTO with optional user confirmation).
        
        Trains model, uploads to MinIO, saves to SQLite, cleans up temp files.
        
        Args:
            interactive: Enable user confirmation for saving?
            
        Returns:
            True if successful
        """
        if not self.person_name:
            raise PipelineError("Phase 1 must be completed first")
        
        self.current_phase = 5
        
        logger.info("\n%s", "="*70)
        logger.info("  PHASE 5: RECOGNITION TRAINING")
        logger.info("%s", "="*70)
        
        self._log_progress(5, "Training face recognition model...")
        
        try:
            # Input paths
            split_dir = get_person_split_dir(self.person_name)
            train_npz = split_dir / "train" / "train.npz"
            test_npz = split_dir / "test" / "test.npz"
            
            if not train_npz.exists():
                raise PipelineError(f"Train NPZ not found: {train_npz}")
            if not test_npz.exists():
                raise PipelineError(f"Test NPZ not found: {test_npz}")
            
            service = RecognitionService(
                storage=self.storage,
                person_name=self.person_name
            )
            
            recognizer, results, metrics, tuning = service.run(
                train_npz_path=str(train_npz),
                test_npz_path=str(test_npz),
                interactive=interactive
            )
            
            if recognizer is None:
                logger.warning("Training cancelled by user or quality gate")
                self._log_progress(5, "Cancelled or rejected")
                return False
            
            logger.info("\n✓ Phase 5 Complete:")
            logger.info("  Accuracy: %.2f%%", metrics.accuracy * 100)
            logger.info("  Uploaded to MinIO: ✓")
            logger.info("  Saved to SQLite: ✓")
            
            self._log_progress(5, f"Training complete: {metrics.accuracy*100:.1f}% accuracy")
            return True
            
        except Exception as ex:
            logger.exception("Phase 5 failed: %s", ex)
            self._log_progress(5, f"FAILED: {ex}")
            raise PipelineError(f"Recognition training failed: {ex}")
    
    # ==================== MAIN WORKFLOW ====================
    
    def run_complete_pipeline(
        self,
        person_name: str,
        auto_reject_threshold: float = 0.6,
        target_frames: int = 100,
        interactive: bool = True
    ) -> bool:
        """
        Run complete pipeline (Phase 1-5).
        
        Phase 1: Manual (user records videos)
        Phase 2-5: Automatic
        
        Args:
            person_name: Person name
            auto_reject_threshold: Quality auto-reject threshold
            target_frames: Target frames per video
            interactive: Enable user confirmation in Phase 5?
            
        Returns:
            True if all phases successful
        """
        logger.info("\n%s", "="*70)
        logger.info("  COMPLETE FACE RECOGNITION PIPELINE")
        logger.info("  Person: %s", person_name)
        logger.info("%s", "="*70)
        
        self._log_progress(0, f"Starting pipeline for: {person_name}")
        
        try:
            # Phase 1: Data Collection (MANUAL)
            if not self.run_phase1_data_collection(
                person_name=person_name,
                auto_reject_threshold=auto_reject_threshold
            ):
                logger.error("Pipeline aborted: Phase 1 failed")
                return False
            
            # Phase 2: Frame Extraction (AUTO)
            if not self.run_phase2_frame_extraction(target_frames=target_frames):
                logger.error("Pipeline aborted: Phase 2 failed")
                return False
            
            # Phase 3: Embedding Generation (AUTO)
            if not self.run_phase3_embedding_generation():
                logger.error("Pipeline aborted: Phase 3 failed")
                return False
            
            # Phase 4: Data Sanitization (AUTO)
            if not self.run_phase4_data_sanitization():
                logger.error("Pipeline aborted: Phase 4 failed")
                return False
            
            # Phase 5: Recognition Training (AUTO with optional confirmation)
            if not self.run_phase5_recognition_training(interactive=interactive):
                logger.error("Pipeline aborted: Phase 5 failed or cancelled")
                return False
            
            # Success!
            logger.info("\n%s", "="*70)
            logger.info("  🎉 PIPELINE COMPLETE!")
            logger.info("%s", "="*70)
            logger.info("\n  Results:")
            logger.info("    Person: %s", person_name)
            logger.info("    Status: ✓ Trained and deployed")
            logger.info("    Storage: MinIO + SQLite")
            logger.info("    Temp files: %s", "Deleted" if self.auto_cleanup else "Retained")
            logger.info("\n  Next Steps:")
            logger.info("    • Test realtime recognition")
            logger.info("    • Run: python tests/test_realtime_recognition.py")
            logger.info("%s", "="*70)
            
            self._log_progress(0, "Pipeline complete!")
            return True
            
        except PipelineError as ex:
            logger.error("\n%s", "="*70)
            logger.error("  ❌ PIPELINE FAILED")
            logger.error("%s", "="*70)
            logger.error("  Error: %s", ex)
            logger.error("  Phase: %d", self.current_phase)
            logger.error("%s", "="*70)
            
            self._log_progress(0, f"Pipeline failed at Phase {self.current_phase}")
            return False
            
        except Exception as ex:
            logger.exception("Unexpected pipeline error: %s", ex)
            self._log_progress(0, f"Unexpected error: {ex}")
            return False
    
    def cleanup_temp_files(self, person_name: Optional[str] = None):
        """
        Cleanup temp files for a person.
        
        Args:
            person_name: Person name (uses self.person_name if None)
        """
        person_name = person_name or self.person_name
        
        if not person_name:
            logger.warning("No person name specified for cleanup")
            return
        
        try:
            temp_dir = get_person_temp_dir(person_name)
            
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info("✓ Cleaned up temp files: %s", temp_dir)
            else:
                logger.debug("Temp dir not found (already cleaned?): %s", temp_dir)
                
        except Exception as ex:
            logger.error("Cleanup failed: %s", ex)


# ==================== CONVENIENCE FUNCTION ====================

def run_pipeline_for_person(
    person_name: str,
    interactive: bool = True,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> bool:
    """
    Convenience function to run complete pipeline for one person.
    
    Args:
        person_name: Person name
        interactive: Enable user confirmation?
        progress_callback: Progress callback function
        
    Returns:
        True if successful
        
    Example:
        success = run_pipeline_for_person("Nghi", interactive=True)
    """
    service = MasterWorkflowService(
        progress_callback=progress_callback,
        auto_cleanup=True
    )
    
    return service.run_complete_pipeline(
        person_name=person_name,
        interactive=interactive
    )


__all__ = ["MasterWorkflowService", "PipelineError", "run_pipeline_for_person"]
