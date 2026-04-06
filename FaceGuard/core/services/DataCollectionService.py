"""
Data Collection Service - Production-ready wrapper.

===================  ARCHITECTURE  ====================
Clean service wrapper around DataCollectionGUI.

Workflow:
1. User enters name
2. GUI records 3 videos (frontal, horizontal, vertical)
3. Each video: Record → Gate 1 check → Playback → Keep/Reject
4. Videos saved to: data/temp/{person}/video/{type}/
5. Return video paths for Phase 2 (Frame Extraction)

Features:
- PySide6 GUI with real-time preview
- Gate 1 quality check integration
- Auto-reject low-quality videos (< 60%)
- Clean, intuitive interface
- Production-ready error handling

===================  REFACTORED 08-03-2026  ====================
- Simplified wrapper API
- Production-ready structure
- Removed test-specific code
- Ready for MasterWorkflowService integration

===================  REFACTORED LOGGING 09-03-2026  ====================
- Enhanced logging with rich + colorama
- Color-coded prefixes for dark theme
- Professional error messages
"""

import logging
from typing import Optional, Dict
from pathlib import Path

from config.settings import INFO, SUCCESS, ERROR, WARNING
from core.adapters.VideoAdapter import VideoAdapter
from core.services.VideoQualityService import VideoQualityService
from core.services.DataCollectionGUI import launch_data_collection_gui

logger = logging.getLogger(__name__)


class DataCollectionService:
    """
    Data collection service with GUI.

    This service orchestrates video recording with integrated
    quality checks (Gate 1) through an intuitive GUI interface.

    Usage:
        # Simple usage
        service = DataCollectionService()
        video_paths = service.collect_videos("Nghi")

        if video_paths:
            # Videos saved to data/temp/Nghi/video/{type}/
            print(f"Ready for Phase 2: {video_paths}")

        # Custom usage
        service = DataCollectionService(auto_reject_threshold=0.7)
        video_paths = service.collect_videos("Nghi")
    """

    def __init__(
        self,
        video_adapter: Optional[VideoAdapter] = None,
        quality_service: Optional[VideoQualityService] = None,
        auto_reject_threshold: float = 0.6
    ):
        """
        Initialize service.

        Args:
            video_adapter: VideoAdapter instance (optional, creates new if None)
            quality_service: VideoQualityService instance (optional, creates new if None)
            auto_reject_threshold: Auto-reject threshold (0-1, default: 0.6)
                Videos with quality < threshold are auto-rejected
        """
        self.video_adapter = video_adapter
        self.quality_service = quality_service
        self.auto_reject_threshold = auto_reject_threshold

    def collect_videos(self, person_name: str) -> Optional[Dict[str, str]]:
        """
        Launch GUI to collect videos for a person.

        This is the main entry point for data collection. It launches
        an interactive GUI that guides the user through recording
        3 videos (frontal, horizontal, vertical) with quality checks.

        Args:
            person_name: Person name (used for folder creation)

        Returns:
            Dict mapping video_type → file_path if successful:
            {
                'frontal': 'data/temp/Nghi/video/frontal/video_001.mp4',
                'horizontal': 'data/temp/Nghi/video/horizontal/video_001.mp4',
                'vertical': 'data/temp/Nghi/video/vertical/video_001.mp4'
            }

            None if cancelled or failed

        Raises:
            ValueError: If person_name is empty
            RuntimeError: If camera cannot be opened
        """
        if not person_name or not person_name.strip():
            raise ValueError("Person name cannot be empty")

        person_name = person_name.strip()
        logger.info(f"{INFO} Starting data collection for person: {person_name}")

        # ✅ FIX: Create and test camera ONCE, then reuse
        video_adapter_to_use = self.video_adapter

        if video_adapter_to_use is None:
            logger.info(f"{INFO} Creating VideoAdapter for camera")
            video_adapter_to_use = VideoAdapter()

            # Open and test camera
            if not video_adapter_to_use.open_camera():
                logger.error(f"{ERROR} Camera not available")
                logger.error(f"{ERROR} Troubleshooting steps:")
                logger.error(f"{ERROR}   1. Check if camera is connected")
                logger.error(f"{ERROR}   2. Close other apps using camera (Zoom, Teams, Skype)")
                logger.error(f"{ERROR}   3. Check Windows Privacy Settings: Camera permissions")
                logger.error(f"{ERROR}   4. Try restarting your computer")
                raise RuntimeError("Cannot open camera - check camera availability")

            logger.info(f"{SUCCESS} Camera opened successfully")
        else:
            # Use provided adapter, ensure it's open
            if not video_adapter_to_use.is_open():
                if not video_adapter_to_use.open_camera():
                    raise RuntimeError("Cannot open camera")
            logger.info(f"{INFO} Using provided VideoAdapter")

        video_paths = launch_data_collection_gui(
            person_name=person_name,
            video_adapter=video_adapter_to_use,  # ✅ Pass opened adapter!
            quality_service=self.quality_service,
            auto_reject_threshold=self.auto_reject_threshold
        )

        if video_paths:
            logger.info(f"{SUCCESS} Data collection completed: {len(video_paths)}/3 videos saved")
            for vtype, path in video_paths.items():
                logger.info(f"{INFO}   {vtype}: {path}")
        else:
            logger.warning(f"{WARNING} Data collection cancelled or failed")

        # ✅ FIX: Clean up camera if we created it
        if self.video_adapter is None and video_adapter_to_use is not None:
            if video_adapter_to_use.is_open():
                video_adapter_to_use.close()
                logger.info(f"{SUCCESS} Camera closed successfully")

        return video_paths


def collect_videos(
    person_name: str,
    auto_reject_threshold: float = 0.6
) -> Optional[Dict[str, str]]:
    """
    Convenience function for quick data collection.

    Args:
        person_name: Person name
        auto_reject_threshold: Auto-reject threshold (0-1)

    Returns:
        Dict of video paths or None

    Example:
        from core.services import collect_videos

        videos = collect_videos("Nghi")
        if videos:
            print("Ready for Phase 2!")
    """
    service = DataCollectionService(auto_reject_threshold=auto_reject_threshold)
    return service.collect_videos(person_name)


__all__ = ["DataCollectionService", "collect_videos"]

