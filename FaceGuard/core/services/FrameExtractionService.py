"""Stateless Frame Extraction Service.

Migrated từ pipeline/Face_Extract_Frames.py - Logic giữ nguyên 100%.

===================  LỊCH SỬ REFACTOR  ====================
Refactored 17-01-2026:
- Loại bỏ: IMAGE_DIRS, METADATA_DIRS, PROCESSED_VIDEO_DIRS imports (đã xóa)
- Migrate: TOÀN BỘ logic từ Face_Extract_Frames.py
  * LightweightFaceDetector (Haar Cascade)
  * QualityAnalyzer (blur, brightness, contrast, exposure)
  * BaseVideoProcessor (abstract base với adaptive thresholding)
  * 3 Specialized Processors (Frontal, Horizontal, Vertical)
- Service: Stateless orchestrator, delegate to processors
- BatchVideoProcessor: KHÔNG migrate (thuộc orchestration layer)

Refactored 31-01-2026:
- ADD SNR CHECK: Frame-level SNR validation (align với Gate 1)
- ADD SMART RANKING: Multi-factor score (blur + face + exposure + SNR)
- ALIGN THRESHOLDS: Derive từ VIDEO_QUALITY_CONFIG (consistency!)
- Mục đích: Better quality, Gate 1 compatibility

Refactored 05-03-2026:
- GỠ BỎ MEDIAPIPE: Chỉ dùng Haar Cascade (đơn giản, ổn định)
- Lý do: MediaPipe 0.10.x phức tạp, Haar đủ dùng cho Gate 2
- Phase 3 sẽ dùng SCRFD (95% accuracy) cho embedding

===================  REFACTORED LOGGING 09-03-2026  ====================
- Enhanced logging with rich + colorama
- Rich tables for comprehensive frame statistics
- Color-coded quality metrics (PASS/FAIL)
- Professional dark theme optimized
- Removed emojis for professional output

Architecture:
- Service = orchestrator only
- Processors = core logic (face detection, quality analysis, bucketing)
- Storage: Service xử lý local temp, caller upload/download
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from config.settings import (
    FRAME_EXTRACTION_CONFIG,
    INFO, SUCCESS, ERROR, WARNING,
    console, create_table, print_section_header,
)
from core.adapters.StorageAdapter import StorageAdapter
from core.entities import ExtractFrameVideoInfo, FrameQuality

logger = logging.getLogger(__name__)

# Load config
CONFIG = {
    'enable_face_detection': FRAME_EXTRACTION_CONFIG['ENABLE_FACE_DETECTION'],
    'min_face_size': FRAME_EXTRACTION_CONFIG['MIN_FACE_SIZE'],
    'require_face_for_extraction': FRAME_EXTRACTION_CONFIG.get('REQUIRE_FACE_FOR_EXTRACTION', False),
    'enable_exposure_check': FRAME_EXTRACTION_CONFIG['ENABLE_EXPOSURE_CHECK'],
    'exposure_clip_threshold': FRAME_EXTRACTION_CONFIG['EXPOSURE_CLIP_THRESHOLD'],
    'enable_frame_snr_check': FRAME_EXTRACTION_CONFIG['ENABLE_FRAME_SNR_CHECK'],
    'frame_min_snr': FRAME_EXTRACTION_CONFIG['FRAME_MIN_SNR'],
    'save_metadata': FRAME_EXTRACTION_CONFIG['SAVE_METADATA'],
    'frontal_blur_threshold': FRAME_EXTRACTION_CONFIG['FRONTAL_BLUR_THRESHOLD'],
    'horizontal_blur_threshold': FRAME_EXTRACTION_CONFIG['HORIZONTAL_BLUR_THRESHOLD'],
    'vertical_blur_threshold': FRAME_EXTRACTION_CONFIG['VERTICAL_BLUR_THRESHOLD'],
    'frontal_brightness_range': FRAME_EXTRACTION_CONFIG['FRONTAL_BRIGHTNESS_RANGE'],
    'horizontal_brightness_range': FRAME_EXTRACTION_CONFIG['HORIZONTAL_BRIGHTNESS_RANGE'],
    'vertical_brightness_range': FRAME_EXTRACTION_CONFIG['VERTICAL_BRIGHTNESS_RANGE'],
    'horizontal_max_per_bucket': FRAME_EXTRACTION_CONFIG['HORIZONTAL_MAX_PER_BUCKET'],
    'vertical_max_per_bucket': FRAME_EXTRACTION_CONFIG['VERTICAL_MAX_PER_BUCKET'],
    'horizontal_angle_bounds': FRAME_EXTRACTION_CONFIG['HORIZONTAL_ANGLE_BOUNDS'],
    'vertical_progress_splits': FRAME_EXTRACTION_CONFIG['VERTICAL_PROGRESS_SPLITS'],
    'sample_interval_min': FRAME_EXTRACTION_CONFIG['SAMPLE_INTERVAL_MIN'],
    'sample_interval_denom': FRAME_EXTRACTION_CONFIG['SAMPLE_INTERVAL_DENOM'],
    'min_select_interval_base': FRAME_EXTRACTION_CONFIG['MIN_SELECT_INTERVAL_BASE'],
    'enable_smart_ranking': FRAME_EXTRACTION_CONFIG['ENABLE_SMART_RANKING'],
    'ranking_weights': FRAME_EXTRACTION_CONFIG['RANKING_WEIGHTS'],
}


# ==================== HAAR FACE DETECTOR ====================
class HaarFaceDetector:
    """
    Simple face detector using Haar Cascade.

    Refactored 05-03-2026:
    - Simplified from ModernFaceDetector (removed MediaPipe)
    - Haar Cascade only (72% accuracy - sufficient for Gate 2)
    - Phase 3 will use SCRFD (95% accuracy) for embedding

    Benefits:
    - Simple and stable
    - No external dependencies (built-in OpenCV)
    - Fast inference
    - Sufficient for frame filtering
    """

    def __init__(self):
        """Initialize Haar Cascade face detector."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)

            if self.detector.empty():
                raise ValueError("Failed to load Haar Cascade")

            logger.info(f"{SUCCESS} Haar Cascade face detector initialized")
            self.available = True

        except Exception as ex:
            logger.error(f"{ERROR} Haar Cascade initialization failed: {ex}")
            logger.warning(f"{WARNING} Proceeding without face detection")
            self.available = False

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces using Haar Cascade.

        Args:
            image: Input image (BGR)

        Returns:
            List of detected faces with:
            - bbox: [x1, y1, x2, y2]
            - confidence: 1.0 (Haar doesn't provide real confidence)
            - width, height: Face size
            - landmarks: [] (empty - Haar doesn't provide landmarks)
        """
        if not self.available:
            return []

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Convert to standard format
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    'bbox': [int(x), int(y), int(x+w), int(y+h)],
                    'confidence': 1.0,  # Haar doesn't provide real confidence
                    'width': int(w),
                    'height': int(h),
                    'landmarks': []  # No landmarks
                })

            return results

        except Exception as ex:
            logger.warning(f"{WARNING} Haar detection error: {ex}")
            return []


# ==================== QUALITY ANALYZER (Enhanced!) ====================
class QualityAnalyzer:
    """
    Enhanced quality analyzer with SNR check.

    Refactored 31-01-2026:
    - ADD SNR calculation: Frame-level SNR (align with Gate 1)
    - Reuse Gate 1 SNR logic (traditional + frequency-based)
    """

    @staticmethod
    def calculate_blur_score(image: np.ndarray) -> float:
        """Laplacian variance blur score."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Average brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.mean()

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Standard deviation as contrast measure."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.std()

    @staticmethod
    def calculate_snr(image: np.ndarray) -> Tuple[float, float]:
        """
        Calculate SNR using SAME logic as Gate 1 (VideoOutlierDetector).

        Refactored 31-01-2026:
        - Reuse Gate 1 SNR calculation
        - Ensure consistency across gates

        Returns:
            (snr_traditional, snr_frequency)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Method 1: Traditional SNR (from Gate 1)
        signal = gray.mean()
        noise_std = gray.std()
        if noise_std < 1.0:
            noise_std = 1.0
        snr_trad = 20 * np.log10(signal / noise_std)

        # Method 2: Frequency-based SNR (from Gate 1)
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        high_freq = cv2.filter2D(gray, -1, kernel)

        noise_energy = np.sqrt(np.mean(high_freq ** 2))
        signal_energy = np.sqrt(np.mean(gray ** 2))

        if noise_energy < 1.0:
            noise_energy = 1.0

        snr_freq = 20 * np.log10(signal_energy / noise_energy)

        return (snr_trad, snr_freq)

    @staticmethod
    def check_exposure(image: np.ndarray, clip_threshold: float = 0.05) -> int:
        """
        Check exposure quality using histogram clipping.

        Args:
            image: Input image (BGR)
            clip_threshold: Max fraction of clipped pixels

        Returns:
            0 = underexposed
            1 = good exposure
            2 = overexposed
        """
        if not CONFIG['enable_exposure_check']:
            return 1  # Assume good if check disabled

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            total_pixels = gray.shape[0] * gray.shape[1]

            # Count very dark pixels (0-5)
            dark_pixels = np.sum(gray < 5) / total_pixels

            # Count very bright pixels (250-255)
            bright_pixels = np.sum(gray > 250) / total_pixels

            # Determine exposure status
            if dark_pixels > clip_threshold:
                return 0  # Underexposed
            elif bright_pixels > clip_threshold:
                return 2  # Overexposed
            else:
                return 1  # Good exposure

        except Exception as ex:
            logger.warning(f"{WARNING} Exposure check error: {ex}")
            return 1  # Default to good


# ==================== BASE VIDEO PROCESSOR ====================
class BaseVideoProcessor(ABC):
    """
    Enhanced base class with Haar Cascade, SNR check, and smart ranking.

    Refactored 05-03-2026:
    - SIMPLIFIED: Chỉ dùng Haar Cascade (removed MediaPipe complexity)
    - KEPT: Frame-level SNR check (align with Gate 1)
    - KEPT: Smart multi-factor ranking (blur + face + exposure + SNR)
    - ALIGN: Thresholds derived from VIDEO_QUALITY_CONFIG

    Migrated từ Face_Extract_Frames.py - Core logic giữ nguyên.

    Architecture:
    - Abstract base class cho 3 specialized processors
    - Common logic: load video, analyze quality, detect fast motion, extract/select frames
    - Subclass implement: should_keep_frame(), get_processor_name()
    """

    def __init__(self, video_path: str, output_dir: str, person_name: str, video_type: str):
        self.video_path = video_path
        self.output_dir = output_dir
        self.person_name = person_name
        self.video_type = video_type
        self.quality_analyzer = QualityAnalyzer()
        self.video_info = None

        # Initialize face detector if enabled
        if CONFIG['enable_face_detection']:
            self.face_detector = HaarFaceDetector()
        else:
            self.face_detector = None
    
    def load_video(self) -> cv2.VideoCapture:
        """Load video và extract metadata."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        self.video_info = ExtractFrameVideoInfo(
            path=self.video_path,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            width=width,
            height=height
        )
        
        return cap
    
    def analyze_frame_quality(self, frame: np.ndarray, frame_num: int) -> FrameQuality:
        """
        Comprehensive frame quality analysis.

        Refactored 04-03-2026:
        - Add SNR calculation (align with Gate 1)
        - Use Haar Cascade (simple detection)
        - No landmarks (Haar doesn't support)

        Args:
            frame: Input frame (BGR)
            frame_num: Frame number

        Returns:
            FrameQuality object with all metrics
        """
        # Calculate timestamp
        timestamp = frame_num / self.video_info.fps if self.video_info.fps > 0 else 0

        # Basic quality metrics
        blur_score = self.quality_analyzer.calculate_blur_score(frame)
        brightness = self.quality_analyzer.calculate_brightness(frame)
        contrast = self.quality_analyzer.calculate_contrast(frame)
        exposure = self.quality_analyzer.check_exposure(frame)

        # SNR calculation (Align with Gate 1)
        snr_trad, snr_freq = self.quality_analyzer.calculate_snr(frame)
        snr_combined = max(snr_trad, snr_freq * 0.6)  # Same weight as Gate 1

        # Face detection (Haar Cascade)
        has_face = False
        face_bbox = [0, 0, 0, 0]
        face_size = (0, 0)
        face_confidence = 0.0

        if self.face_detector and self.face_detector.available:
            faces = self.face_detector.detect(frame)
            if faces:
                # Select largest face
                largest_face = max(faces, key=lambda f: f['width'] * f['height'])
                has_face = True
                face_bbox = largest_face['bbox']
                face_size = (largest_face['width'], largest_face['height'])
                face_confidence = largest_face['confidence']  # Always 1.0 for Haar

        return FrameQuality(
            frame_num=frame_num,
            timestamp=timestamp,
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            exposure_status=exposure,
            has_face=has_face,
            face_bbox=face_bbox,
            face_size=face_size,
            face_confidence=face_confidence,
            snr=snr_combined,
            snr_frequency=snr_freq,
            frame_data=frame.copy()
        )
    
    def _is_frame_high_quality(
        self,
        frame_quality: FrameQuality,
        blur_threshold: float,
        brightness_range: Tuple[float, float],
        require_face: bool,
        min_face_size: int,
    ) -> bool:
        """
        Evaluate frame quality (outside entity to keep entities data-only).

        Refactored 04-03-2026:
        - Add SNR check
        - Remove face confidence check (Haar always returns 1.0)
        """
        # Basic quality checks
        basic_quality = (
            frame_quality.blur_score > blur_threshold
            and brightness_range[0] < frame_quality.brightness < brightness_range[1]
            and frame_quality.exposure_status == 1
        )

        if not basic_quality:
            return False

        # SNR check (Align with Gate 1)
        if CONFIG['enable_frame_snr_check']:
            if frame_quality.snr < CONFIG['frame_min_snr']:
                return False  # Reject noisy frames

        # Face checks (simpler without confidence)
        if require_face:
            if not frame_quality.has_face:
                return False
            if min_face_size > 0 and min(frame_quality.face_size) < min_face_size:
                return False

        return True
    
    @abstractmethod
    def should_keep_frame(self, frame_quality: FrameQuality, frame_num: int) -> bool:
        """Quyết định có giữ frame này không."""
        pass
    
    @abstractmethod
    def get_processor_name(self) -> str:
        """Tên processor."""
        pass
    
    def _detect_fast_motion(self, cap: cv2.VideoCapture) -> bool:
        """Fast motion detection."""
        sample_count = min(30, self.video_info.total_frames)
        blur_diffs = []
        
        prev_blur = None
        for i in range(sample_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if prev_blur is not None:
                blur_diffs.append(abs(blur - prev_blur))
            prev_blur = blur
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if not blur_diffs:
            return False
        
        blur_variation = np.std(blur_diffs)
        mean_diff = np.mean(blur_diffs)
        is_fast_motion = blur_variation > 20 or mean_diff > 30

        if is_fast_motion:
            logger.info(f"{INFO} Fast motion detected")
            logger.info(f"    Blur variation: {blur_variation:.1f}")
            logger.info(f"    Mean diff: {mean_diff:.1f}")

        return is_fast_motion
    
    def _analyze_video_quality(self, cap: cv2.VideoCapture) -> Dict:
        """Video quality analysis."""
        sample_count = min(50, self.video_info.total_frames)
        blur_scores = []
        brightness_values = []
        
        for i in range(sample_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            bright = gray.mean()
            
            blur_scores.append(blur)
            brightness_values.append(bright)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        p20_blur = np.percentile(blur_scores, 20)
        p10_blur = np.percentile(blur_scores, 10)
        adaptive_threshold = max(40, p20_blur * 0.75)
        
        if np.mean(blur_scores) < 60 and np.std(blur_scores) < 15:
            adaptive_threshold = max(35, p10_blur * 0.7)
            logger.info(f"{INFO} Low consistent blur detected - adjusting threshold")

        return {
            'mean_blur': np.mean(blur_scores),
            'std_blur': np.std(blur_scores),
            'p20_blur': p20_blur,
            'p10_blur': p10_blur,
            'mean_brightness': np.mean(brightness_values),
            'adaptive_threshold': adaptive_threshold
        }
    
    def extract_frames(self, target_frames: int = 100) -> List[str]:
        """
        Main extraction method with enhanced logging.

        Args:
            target_frames: Target number of frames to extract

        Returns:
            List of saved frame paths
        """
        print_section_header(f"{self.get_processor_name()}")
        logger.info(f"{INFO} Video: {os.path.basename(self.video_path)}")

        cap = self.load_video()

        # Video Info Table
        video_info_table = create_table(
            "Video Information",
            ["Property", "Value"],
            [
                ["Path", self.video_info.path],
                ["Duration", f"{self.video_info.duration:.1f}s"],
                ["FPS", f"{self.video_info.fps:.1f}"],
                ["Resolution", f"{self.video_info.width}x{self.video_info.height}"],
                ["Total Frames", f"{self.video_info.total_frames}"],
            ]
        )
        console.print(video_info_table)

        # Feature Configuration Table
        feature_table = create_table(
            "Feature Configuration",
            ["Feature", "Status", "Details"],
            [
                ["Face Detection", 
                 "Enabled" if CONFIG['enable_face_detection'] else "Disabled",
                 "Haar Cascade" if CONFIG['enable_face_detection'] else "N/A"],
                ["Exposure Check", 
                 "Enabled" if CONFIG['enable_exposure_check'] else "Disabled",
                 "Histogram Clipping" if CONFIG['enable_exposure_check'] else "N/A"],
                ["SNR Check", 
                 "Enabled" if CONFIG['enable_frame_snr_check'] else "Disabled",
                 f"Min: {CONFIG['frame_min_snr']:.0f}dB" if CONFIG['enable_frame_snr_check'] else "N/A"],
                ["Smart Ranking", 
                 "Enabled" if CONFIG['enable_smart_ranking'] else "Disabled",
                 "Multi-factor" if CONFIG['enable_smart_ranking'] else "Blur-only"],
                ["Metadata Export", 
                 "Enabled" if CONFIG['save_metadata'] else "Disabled",
                 "CSV format" if CONFIG['save_metadata'] else "N/A"],
            ]
        )
        console.print(feature_table)

        is_fast_motion = self._detect_fast_motion(cap)
        video_quality = self._analyze_video_quality(cap)
        adaptive_threshold = video_quality['adaptive_threshold']

        # Quality Analysis Table
        quality_table = create_table(
            "Video Quality Analysis",
            ["Metric", "Value", "Status"],
            [
                ["Mean Blur", f"{video_quality['mean_blur']:.1f}", ""],
                ["Std Blur", f"{video_quality['std_blur']:.1f}", ""],
                ["P20 Blur", f"{video_quality['p20_blur']:.1f}", ""],
                ["Adaptive Threshold", f"{adaptive_threshold:.1f}", 
                 "Reduced" if is_fast_motion else "Normal"],
                ["Fast Motion", "Yes" if is_fast_motion else "No", 
                 "WARNING" if is_fast_motion else "OK"],
            ]
        )
        console.print(quality_table)

        if is_fast_motion:
            adaptive_threshold *= 0.9
            logger.warning(f"{WARNING} Fast motion detected - reduced threshold to {adaptive_threshold:.1f}")

        original_threshold = self.blur_threshold
        if adaptive_threshold < self.blur_threshold:
            logger.warning(f"{WARNING} Adjusted threshold: {self.blur_threshold} -> {adaptive_threshold:.0f}")
            self.blur_threshold = adaptive_threshold
        else:
            logger.info(f"{INFO} Keeping original threshold: {self.blur_threshold}")

        logger.info(f"\n{INFO} Step 1: Extracting candidate frames...")
        candidates = self._extract_candidates(cap)
        cap.release()

        logger.info(f"{SUCCESS} Found {len(candidates)} candidate frames")

        # Face detection statistics
        if CONFIG['enable_face_detection'] and candidates:
            faces_detected = sum(1 for c in candidates if c.has_face)
            logger.info(f"{INFO} Faces detected: {faces_detected}/{len(candidates)} ({faces_detected/len(candidates)*100:.1f}%)")
            if faces_detected > 0:
                avg_confidence = np.mean([c.face_confidence for c in candidates if c.has_face])
                logger.info(f"{INFO} Average confidence: {avg_confidence:.2f}")

        if len(candidates) == 0:
            logger.error(f"{ERROR} No frames found - extraction failed")
            return []

        logger.info(f"\n{INFO} Step 2: Selecting {target_frames} best frames...")
        selected = self._select_best_frames(candidates, target_frames)
        logger.info(f"{SUCCESS} Selected {len(selected)} frames")

        logger.info(f"\n{INFO} Step 3: Saving frames and metadata...")
        saved_paths = self._save_frames(selected)

        # Export metadata
        if CONFIG['save_metadata']:
            self._save_metadata(selected)

        self._print_statistics(selected)

        return saved_paths
    
    def _extract_candidates(self, cap: cv2.VideoCapture) -> List[FrameQuality]:
        """Extract candidate frames with COMPREHENSIVE logging for ALL frames."""
        candidates = []
        all_frames_metrics = []  # Store ALL frame metrics for analysis
        frame_num = 0

        # Statistics tracking
        rejected_blur = 0
        rejected_brightness = 0
        rejected_exposure = 0
        rejected_snr = 0
        rejected_no_face = 0
        rejected_face_size = 0

        logger.info("    ═══════════════════════════════════════════════════")
        logger.info("    ANALYZING ALL FRAMES (Showing first 50 + samples)")
        logger.info("    ═══════════════════════════════════════════════════")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_quality = self.analyze_frame_quality(frame, frame_num)

            # Store metrics for ALL frames
            all_frames_metrics.append({
                'frame_num': frame_num,
                'blur': frame_quality.blur_score,
                'brightness': frame_quality.brightness,
                'snr': frame_quality.snr,
                'exposure': frame_quality.exposure_status,
                'has_face': frame_quality.has_face,
                'face_size': min(frame_quality.face_size) if frame_quality.has_face else 0,
            })

            # Show DETAILED metrics for first 50 frames + every 50th frame
            if frame_num < 50 or frame_num % 50 == 0:
                status = "✓ PASS" if self.should_keep_frame(frame_quality, frame_num) else "✗ REJECT"

                logger.info(f"    Frame {frame_num:4d} | {status}")
                logger.info(f"      Blur      : {frame_quality.blur_score:6.1f} (threshold: {self.blur_threshold:5.1f}) {'✓' if frame_quality.blur_score > self.blur_threshold else '✗ FAIL'}")
                logger.info(f"      Brightness: {frame_quality.brightness:6.1f} (range: {self.brightness_range[0]:.0f}-{self.brightness_range[1]:.0f}) {'✓' if self.brightness_range[0] < frame_quality.brightness < self.brightness_range[1] else '✗ FAIL'}")
                logger.info(f"      Exposure  : {frame_quality.exposure_status} (1=good) {'✓' if frame_quality.exposure_status == 1 else '✗ FAIL'}")
                logger.info(f"      SNR       : {frame_quality.snr:6.1f} dB (min: {CONFIG['frame_min_snr']:.1f}) {'✓' if frame_quality.snr >= CONFIG['frame_min_snr'] else '✗ FAIL'}")

                if CONFIG['enable_face_detection']:
                    face_info = f"{'Yes' if frame_quality.has_face else 'No'}"
                    if frame_quality.has_face:
                        face_info += f" (size: {frame_quality.face_size[0]}x{frame_quality.face_size[1]})"
                        face_check = "✓" if min(frame_quality.face_size) >= CONFIG['min_face_size'] else "✗ FAIL"
                    else:
                        face_check = "✗ FAIL" if CONFIG.get('require_face_for_extraction', False) else "✓"
                    logger.info(f"      Face      : {face_info} {face_check}")
                logger.info("")  # Blank line for readability

            # Track rejection reasons
            is_kept = self.should_keep_frame(frame_quality, frame_num)

            if is_kept:
                candidates.append(frame_quality)
            else:
                # Detailed rejection tracking
                if frame_quality.blur_score <= self.blur_threshold:
                    rejected_blur += 1
                if not (self.brightness_range[0] < frame_quality.brightness < self.brightness_range[1]):
                    rejected_brightness += 1
                if frame_quality.exposure_status != 1:
                    rejected_exposure += 1
                if CONFIG['enable_frame_snr_check'] and frame_quality.snr < CONFIG['frame_min_snr']:
                    rejected_snr += 1

                require_face = CONFIG['enable_face_detection'] and CONFIG.get('require_face_for_extraction', False)
                if require_face:
                    if not frame_quality.has_face:
                        rejected_no_face += 1
                    elif min(frame_quality.face_size) < CONFIG['min_face_size']:
                        rejected_face_size += 1

            frame_num += 1

            if frame_num % 30 == 0:
                progress = (frame_num / self.video_info.total_frames) * 100
                logger.info("    Progress: %.0f%% - Candidates: %d/%d", progress, len(candidates), frame_num)

        # ═══════════════════════════════════════════════════════
        # COMPREHENSIVE STATISTICS FOR ALL FRAMES
        # ═══════════════════════════════════════════════════════

        logger.info("\n")
        logger.info("    ═══════════════════════════════════════════════════")
        logger.info("    COMPREHENSIVE FRAME QUALITY ANALYSIS")
        logger.info("    ═══════════════════════════════════════════════════")

        total_processed = frame_num
        total_rejected = total_processed - len(candidates)

        # Extract metrics arrays
        blur_scores = [m['blur'] for m in all_frames_metrics]
        brightness_values = [m['brightness'] for m in all_frames_metrics]
        snr_values = [m['snr'] for m in all_frames_metrics]

        logger.info(f"\n{INFO} OVERALL STATISTICS:")
        logger.info(f"    Total frames    : {total_processed}")
        logger.info(f"    Accepted        : {len(candidates)} ({len(candidates)/total_processed*100:.1f}%)")
        logger.info(f"    Rejected        : {total_rejected} ({total_rejected/total_processed*100:.1f}%)")

        logger.info(f"\n{INFO} BLUR SCORE DISTRIBUTION:")
        logger.info(f"    Threshold       : {self.blur_threshold:.1f}")
        logger.info(f"    Mean            : {np.mean(blur_scores):.1f}")
        logger.info(f"    Median          : {np.median(blur_scores):.1f}")
        logger.info(f"    Std Dev         : {np.std(blur_scores):.1f}")
        logger.info(f"    Min/Max         : {np.min(blur_scores):.1f} / {np.max(blur_scores):.1f}")
        logger.info(f"    Percentiles:")
        logger.info(f"      P10           : {np.percentile(blur_scores, 10):.1f}")
        logger.info(f"      P25           : {np.percentile(blur_scores, 25):.1f}")
        logger.info(f"      P50 (median)  : {np.percentile(blur_scores, 50):.1f}")
        logger.info(f"      P75           : {np.percentile(blur_scores, 75):.1f}")
        logger.info(f"      P90           : {np.percentile(blur_scores, 90):.1f}")
        above_threshold = sum(1 for b in blur_scores if b > self.blur_threshold)
        logger.info(f"    Above threshold : {above_threshold}/{total_processed} ({above_threshold/total_processed*100:.1f}%)")

        logger.info(f"\n{INFO} BRIGHTNESS DISTRIBUTION:")
        logger.info(f"    Range           : {self.brightness_range[0]:.0f} - {self.brightness_range[1]:.0f}")
        logger.info(f"    Mean            : {np.mean(brightness_values):.1f}")
        logger.info(f"    Median          : {np.median(brightness_values):.1f}")
        logger.info(f"    Std Dev         : {np.std(brightness_values):.1f}")
        logger.info(f"    Min/Max         : {np.min(brightness_values):.1f} / {np.max(brightness_values):.1f}")
        in_range = sum(1 for b in brightness_values if self.brightness_range[0] < b < self.brightness_range[1])
        logger.info(f"    In range        : {in_range}/{total_processed} ({in_range/total_processed*100:.1f}%)")

        logger.info(f"\n{INFO} SNR DISTRIBUTION:")
        logger.info(f"    Threshold       : {CONFIG['frame_min_snr']:.1f} dB")
        logger.info(f"    Mean            : {np.mean(snr_values):.1f} dB")
        logger.info(f"    Median          : {np.median(snr_values):.1f} dB")
        logger.info(f"    Std Dev         : {np.std(snr_values):.1f} dB")
        logger.info(f"    Min/Max         : {np.min(snr_values):.1f} / {np.max(snr_values):.1f} dB")
        above_snr = sum(1 for s in snr_values if s >= CONFIG['frame_min_snr'])
        logger.info(f"    Above threshold : {above_snr}/{total_processed} ({above_snr/total_processed*100:.1f}%)")

        if CONFIG['enable_face_detection']:
            faces_detected = sum(1 for m in all_frames_metrics if m['has_face'])
            logger.info(f"\n{INFO} FACE DETECTION:")
            logger.info(f"    Faces found     : {faces_detected}/{total_processed} ({faces_detected/total_processed*100:.1f}%)")
            if faces_detected > 0:
                face_sizes = [m['face_size'] for m in all_frames_metrics if m['has_face']]
                logger.info(f"    Mean face size  : {np.mean(face_sizes):.0f} px")
                logger.info(f"    Min face size   : {CONFIG['min_face_size']} px (threshold)")
                above_face_size = sum(1 for s in face_sizes if s >= CONFIG['min_face_size'])
                logger.info(f"    Above threshold : {above_face_size}/{faces_detected} ({above_face_size/faces_detected*100:.1f}%)")

        if total_rejected > 0:
            logger.info(f"\n{WARNING} REJECTION BREAKDOWN:")
            if rejected_blur > 0:
                logger.info(f"    Blur too low    : {rejected_blur} ({rejected_blur/total_rejected*100:.1f}%)")
            if rejected_brightness > 0:
                logger.info(f"    Brightness OOR  : {rejected_brightness} ({rejected_brightness/total_rejected*100:.1f}%)")
            if rejected_exposure > 0:
                logger.info(f"    Bad exposure    : {rejected_exposure} ({rejected_exposure/total_rejected*100:.1f}%)")
            if rejected_snr > 0:
                logger.info(f"    SNR too low     : {rejected_snr} ({rejected_snr/total_rejected*100:.1f}%)")
            if rejected_no_face > 0:
                logger.info(f"    No face         : {rejected_no_face} ({rejected_no_face/total_rejected*100:.1f}%)")
            if rejected_face_size > 0:
                logger.info(f"    Face too small  : {rejected_face_size} ({rejected_face_size/total_rejected*100:.1f}%)")

        logger.info("\n    ═══════════════════════════════════════════════════")

        # Export detailed metrics to CSV for further analysis
        self._export_all_frames_metrics(all_frames_metrics)

        return candidates
    
    def _select_best_frames(self, candidates: List[FrameQuality], target: int) -> List[FrameQuality]:
        """
        Select best frames using smart ranking.

        Refactored 31-01-2026:
        - ADD smart ranking: Multi-factor score (blur + face + exposure + SNR)
        - Replace blur-only ranking
        """
        if len(candidates) <= target:
            return candidates

        # Smart ranking (NEW!)
        if CONFIG['enable_smart_ranking']:
            sorted_candidates = sorted(
                candidates, 
                key=lambda x: self._calculate_quality_score(x), 
                reverse=True
            )
        else:
            # Fallback to blur-only
            sorted_candidates = sorted(candidates, key=lambda x: x.blur_score, reverse=True)

        min_interval = max(CONFIG['min_select_interval_base'], self.video_info.total_frames // (target * 2))

        selected = []
        for candidate in sorted_candidates:
            is_far_enough = all(
                abs(candidate.frame_num - sel.frame_num) >= min_interval
                for sel in selected
            )

            if is_far_enough:
                selected.append(candidate)

            if len(selected) >= target:
                break

        return sorted(selected, key=lambda x: x.frame_num)

    def _calculate_quality_score(self, frame_quality: FrameQuality) -> float:
        """
        Calculate composite quality score.

        NEW in 31-01-2026:
        - Multi-factor ranking: blur + face + exposure + SNR
        - Weights tuned for face recognition

        Weights:
        - Blur: 30% (important but not everything)
        - Face: 40% (most important! size × confidence)
        - Exposure: 15%
        - SNR: 15%

        Returns:
            Composite score (0-1, higher better)
        """
        weights = CONFIG['ranking_weights']

        # 1. Blur quality (0-1, higher better)
        # Normalize to 150 (excellent blur score)
        blur_norm = min(frame_quality.blur_score / 150.0, 1.0)

        # 2. Face quality (0-1, higher better)
        if frame_quality.has_face and CONFIG['enable_face_detection']:
            # Face size normalized to 200px
            face_size_norm = min(min(frame_quality.face_size) / 200.0, 1.0)
            # Face confidence from MediaPipe (0-1)
            face_conf = frame_quality.face_confidence
            # Combined face quality
            face_quality = face_size_norm * face_conf
        else:
            face_quality = 0.0

        # 3. Exposure quality (0-1, higher better)
        if frame_quality.exposure_status == 1:  # Good
            exposure_quality = 1.0
        else:
            exposure_quality = 0.0  # Under/over exposed

        # 4. SNR quality (0-1, higher better)
        # Normalize to 20dB (excellent SNR)
        snr_norm = min(frame_quality.snr / 20.0, 1.0)

        # Composite score
        score = (
            weights['blur'] * blur_norm +
            weights['face'] * face_quality +
            weights['exposure'] * exposure_quality +
            weights['snr'] * snr_norm
        )

        return score
    
    def _save_frames(self, frames: List[FrameQuality]) -> List[str]:
        """Save frames to disk with detailed logging."""
        output_path = self.output_dir
        os.makedirs(output_path, exist_ok=True)

        saved_paths = []
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        logger.info(f"\n{INFO} Saving frames:")
        logger.info(f"    Output dir : {output_path}")
        logger.info(f"    Name prefix: {self.person_name}_{self.video_type}_{video_name}_frame")
        logger.info(f"    Format     : JPEG")
        logger.info(f"    Count      : {len(frames)} frames")

        for idx, frame_quality in enumerate(frames):
            filename = (f"{self.person_name}_{self.video_type}_"
                       f"{video_name}_frame{frame_quality.frame_num:04d}.jpg")
            filepath = os.path.join(output_path, filename)

            cv2.imwrite(filepath, frame_quality.frame_data)
            saved_paths.append(filepath)

            if (idx + 1) % 20 == 0:
                logger.info(f"    Saved: {idx + 1}/{len(frames)}")

        # Final size calculation
        total_size = sum(Path(p).stat().st_size for p in saved_paths) / 1024 / 1024

        logger.info(f"\n{SUCCESS} Saved {len(saved_paths)} frames successfully")
        logger.info(f"  Location: {output_path}")
        logger.info(f"  Total size: {total_size:.2f} MB (avg: {total_size*1024/len(saved_paths):.1f} KB/frame)")
        logger.warning(f"{WARNING} NO MinIO upload (temp files only)")
        logger.warning(f"{WARNING} NO database insert (temp files only)")
        logger.info(f"{SUCCESS} Ready for Phase 3 (Embedding)")

        return saved_paths
    
    def _save_metadata(self, frames: List[FrameQuality]) -> None:
        """
        Export frame metadata to CSV.

        Args:
            frames: List of selected frames
        """
        try:
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]

            # Build metadata CSV path
            metadata_path = os.path.join(
                self.output_dir,
                f"{self.person_name}_{self.video_type}_{video_name}_metadata.csv"
            )

            logger.info(f"\n{INFO} Exporting metadata:")
            logger.info(f"    File       : {os.path.basename(metadata_path)}")
            logger.info(f"    Format     : CSV (UTF-8 with BOM)")
            logger.info(f"    Columns    : 16 fields")

            # Prepare metadata
            metadata_list = []
            for frame in frames:
                metadata_list.append({
                    'frame_num': frame.frame_num,
                    'timestamp': f"{frame.timestamp:.3f}",
                    'blur_score': f"{frame.blur_score:.2f}",
                    'brightness': f"{frame.brightness:.2f}",
                    'contrast': f"{frame.contrast:.2f}",
                    'exposure_status': frame.exposure_status,
                    'has_face': frame.has_face,
                    'face_x1': frame.face_bbox[0],
                    'face_y1': frame.face_bbox[1],
                    'face_x2': frame.face_bbox[2],
                    'face_y2': frame.face_bbox[3],
                    'face_width': frame.face_size[0],
                    'face_height': frame.face_size[1],
                    'face_confidence': f"{frame.face_confidence:.3f}",
                    'snr': f"{frame.snr:.2f}",
                    'snr_frequency': f"{frame.snr_frequency:.2f}",
                    'filename': f"{self.person_name}_{self.video_type}_{video_name}_frame{frame.frame_num:04d}.jpg"
                })

            # Save to CSV
            df = pd.DataFrame(metadata_list)
            df.to_csv(metadata_path, index=False, encoding='utf-8-sig')

            file_size = Path(metadata_path).stat().st_size / 1024

            logger.info(f"{SUCCESS} Metadata exported successfully:")
            logger.info(f"    Rows       : {len(metadata_list)}")
            logger.info(f"    Size       : {file_size:.1f} KB")
            logger.warning(f"{WARNING} NO MinIO upload (temp file only)")
            logger.info(f"{SUCCESS} Ready for analysis/debugging")

        except Exception as ex:
            logger.error(f"{ERROR} Metadata export failed: {ex}")
            logger.exception(f"{ERROR} Stack trace:")

    def _export_all_frames_metrics(self, all_frames_metrics: List[Dict]) -> None:
        """
        Export ALL frames metrics to CSV for analysis (including rejected frames).

        Args:
            all_frames_metrics: List of metrics for ALL frames
        """
        try:
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]

            # Build CSV path
            analysis_path = os.path.join(
                self.output_dir,
                f"{self.person_name}_{self.video_type}_{video_name}_ALL_FRAMES_ANALYSIS.csv"
            )

            # Add pass/fail columns
            for m in all_frames_metrics:
                m['pass_blur'] = m['blur'] > self.blur_threshold
                m['pass_brightness'] = self.brightness_range[0] < m['brightness'] < self.brightness_range[1]
                m['pass_snr'] = m['snr'] >= CONFIG['frame_min_snr']
                m['pass_exposure'] = m['exposure'] == 1
                m['pass_face_size'] = m['face_size'] >= CONFIG['min_face_size'] if m['has_face'] else False
                m['pass_all'] = (m['pass_blur'] and m['pass_brightness'] and 
                                m['pass_snr'] and m['pass_exposure'])

            # Save to CSV
            df = pd.DataFrame(all_frames_metrics)
            df.to_csv(analysis_path, index=False, encoding='utf-8-sig')

            file_size = Path(analysis_path).stat().st_size / 1024

            logger.info(f"\n{INFO} ALL FRAMES ANALYSIS EXPORTED:")
            logger.info(f"    File       : {os.path.basename(analysis_path)}")
            logger.info(f"    Rows       : {len(all_frames_metrics)}")
            logger.info(f"    Size       : {file_size:.1f} KB")
            logger.info(f"    Location   : {self.output_dir}")
            logger.info(f"{SUCCESS} Use this file to tune thresholds")

        except Exception as ex:
            logger.error(f"{ERROR} All frames analysis export failed: {ex}")
            logger.exception(f"{ERROR} Stack trace:")
    
    def _print_statistics(self, frames: List[FrameQuality]):
        """
        Enhanced statistics with threshold comparison and quality distribution.

        Refactored 09-03-2026:
        - Professional logging without emojis
        - Clean metric presentation
        """
        if not frames:
            return

        blur_scores = [f.blur_score for f in frames]
        brightness_values = [f.brightness for f in frames]
        snr_values = [f.snr for f in frames] if CONFIG['enable_frame_snr_check'] else []

        logger.info(f"\n{INFO} Quality Statistics (Selected Frames):")
        logger.info("    " + "="*50)

        # Blur statistics
        logger.info("    Blur score:")
        logger.info(f"      Mean       : {np.mean(blur_scores):.1f} (+/- {np.std(blur_scores):.1f})")
        logger.info(f"      Min/Max    : {min(blur_scores):.1f} / {max(blur_scores):.1f}")
        logger.info(f"      Threshold  : {self.blur_threshold:.1f} (all frames above threshold)")

        # Brightness statistics  
        logger.info("    Brightness:")
        logger.info(f"      Mean       : {np.mean(brightness_values):.1f} (+/- {np.std(brightness_values):.1f})")
        logger.info(f"      Min/Max    : {min(brightness_values):.1f} / {max(brightness_values):.1f}")
        logger.info(f"      Range      : {self.brightness_range[0]:.0f}-{self.brightness_range[1]:.0f} (all frames in range)")

        # Face detection stats (Simplified for Haar)
        if CONFIG['enable_face_detection']:
            faces_count = sum(1 for f in frames if f.has_face)
            if faces_count > 0:
                face_sizes = [min(f.face_size) for f in frames if f.has_face]

                logger.info("    Face detection:")
                logger.info(f"      Found      : {faces_count}/{len(frames)} ({faces_count/len(frames)*100:.1f}%)")
                logger.info(f"      Avg size   : {np.mean(face_sizes):.0f} px (+/- {np.std(face_sizes):.0f})")
                logger.info(f"      Size range : {min(face_sizes):.0f} - {max(face_sizes):.0f} px")
                logger.info(f"      Min size   : {CONFIG['min_face_size']} px (threshold)")

        # Exposure stats
        if CONFIG['enable_exposure_check']:
            good_exposure = sum(1 for f in frames if f.exposure_status == 1)
            under_exposure = sum(1 for f in frames if f.exposure_status == 0)
            over_exposure = sum(1 for f in frames if f.exposure_status == 2)

            logger.info("    Exposure:")
            logger.info(f"      Good       : {good_exposure}/{len(frames)} ({good_exposure/len(frames)*100:.1f}%)")
            if under_exposure > 0:
                logger.info(f"      Under      : {under_exposure}/{len(frames)} ({under_exposure/len(frames)*100:.1f}%)")
            if over_exposure > 0:
                logger.info(f"      Over       : {over_exposure}/{len(frames)} ({over_exposure/len(frames)*100:.1f}%)")

        # SNR stats
        if CONFIG['enable_frame_snr_check'] and snr_values:
            logger.info("    SNR:")
            logger.info(f"      Mean       : {np.mean(snr_values):.1f} dB (+/- {np.std(snr_values):.1f})")
            logger.info(f"      Min/Max    : {min(snr_values):.1f} / {max(snr_values):.1f} dB")
            logger.info(f"      Threshold  : {CONFIG['frame_min_snr']:.1f} dB (all frames above threshold)")

        # Quality distribution
        logger.info(f"\n{INFO} Quality Distribution:")

        # Blur distribution (quartiles)
        q25_blur = np.percentile(blur_scores, 25)
        q50_blur = np.percentile(blur_scores, 50)
        q75_blur = np.percentile(blur_scores, 75)
        logger.info("    Blur quartiles:")
        logger.info(f"      Q1 (25%)   : {q25_blur:.1f}")
        logger.info(f"      Q2 (50%)   : {q50_blur:.1f} (median)")
        logger.info(f"      Q3 (75%)   : {q75_blur:.1f}")

        logger.info("    " + "="*50)




# ==================== SPECIALIZED PROCESSORS ====================
class FrontalExpressionProcessor(BaseVideoProcessor):
    """
    Processor cho frontal videos.
    
    Migrated từ Face_Extract_Frames.py - Logic giữ nguyên 100%.
    """
    
    def __init__(self, video_path: str, output_dir: str, person_name: str):
        super().__init__(video_path, output_dir, person_name, 'frontal')
        self.blur_threshold = CONFIG['frontal_blur_threshold']
        self.brightness_range = CONFIG['frontal_brightness_range']
    
    def should_keep_frame(self, frame_quality: FrameQuality, frame_num: int) -> bool:
        """Selection logic cho frontal frames."""
        require_face = CONFIG['enable_face_detection'] and CONFIG.get('require_face_for_extraction', False)

        if not self._is_frame_high_quality(
            frame_quality,
            self.blur_threshold,
            self.brightness_range,
            require_face,
            CONFIG['min_face_size'],
        ):
            return False

        sample_interval = max(
            CONFIG['sample_interval_min'],
            self.video_info.total_frames // CONFIG['sample_interval_denom']
        )
        return frame_num % sample_interval == 0
    
    def get_processor_name(self) -> str:
        return "FRONTAL_EXPRESSION_PROCESSOR"


class HorizontalRotationProcessor(BaseVideoProcessor):
    """
    Processor cho horizontal rotation videos.
    
    Migrated từ Face_Extract_Frames.py - Logic giữ nguyên 100%.
    """
    
    def __init__(self, video_path: str, output_dir: str, person_name: str):
        super().__init__(video_path, output_dir, person_name, 'horizontal')
        self.blur_threshold = CONFIG['horizontal_blur_threshold']
        self.brightness_range = CONFIG['horizontal_brightness_range']
        self.angle_buckets = {
            'left_extreme': 0,
            'left_semi': 0,
            'frontal': 0,
            'right_semi': 0,
            'right_extreme': 0
        }
        self.max_per_bucket = CONFIG['horizontal_max_per_bucket']
    
    def should_keep_frame(self, frame_quality: FrameQuality, frame_num: int) -> bool:
        """Selection logic với angle bucketing."""
        require_face = CONFIG['enable_face_detection'] and CONFIG.get('require_face_for_extraction', False)

        if not self._is_frame_high_quality(
            frame_quality,
            self.blur_threshold,
            self.brightness_range,
            require_face,
            CONFIG['min_face_size'],
        ):
            return False

        progress = frame_num / self.video_info.total_frames
        estimated_angle = -90 + (progress * 180)
        bucket = self._get_angle_bucket(estimated_angle)

        if self.angle_buckets[bucket] >= self.max_per_bucket:
            return False

        self.angle_buckets[bucket] += 1
        return True
    
    def _get_angle_bucket(self, angle: float) -> str:
        """Determine angle bucket."""
        left_extreme, left_semi, right_semi, right_extreme = CONFIG['horizontal_angle_bounds']
        if angle < left_extreme:
            return 'left_extreme'
        elif angle < left_semi:
            return 'left_semi'
        elif angle < right_semi:
            return 'frontal'
        elif angle < right_semi:
            return 'right_semi'
        else:
            return 'right_extreme'
    
    def get_processor_name(self) -> str:
        return "HORIZONTAL_ROTATION_PROCESSOR"
    
    def _print_statistics(self, frames: List[FrameQuality]):
        """Override với angle distribution."""
        super()._print_statistics(frames)
        
        logger.info("\n  Phân bố góc độ:")
        for bucket, count in self.angle_buckets.items():
            logger.info("    %-15s : %d frames", bucket, count)


class VerticalRollProcessor(BaseVideoProcessor):
    """
    Processor cho vertical movement videos.
    
    Migrated từ Face_Extract_Frames.py - Logic giữ nguyên 100%.
    """
    
    def __init__(self, video_path: str, output_dir: str, person_name: str):
        super().__init__(video_path, output_dir, person_name, 'vertical')
        self.blur_threshold = CONFIG['vertical_blur_threshold']
        self.brightness_range = CONFIG['vertical_brightness_range']
        self.movement_buckets = {
            'look_down': 0,
            'frontal': 0,
            'look_up': 0,
            'roll_left': 0,
            'roll_right': 0
        }
        self.max_per_bucket = CONFIG['vertical_max_per_bucket']
    
    def should_keep_frame(self, frame_quality: FrameQuality, frame_num: int) -> bool:
        """Selection logic với movement bucketing."""
        require_face = CONFIG['enable_face_detection'] and CONFIG.get('require_face_for_extraction', False)

        if not self._is_frame_high_quality(
            frame_quality,
            self.blur_threshold,
            self.brightness_range,
            require_face,
            CONFIG['min_face_size'],
        ):
            return False

        progress = frame_num / self.video_info.total_frames

        split1, split2, split3, split4 = CONFIG['vertical_progress_splits']

        if progress < split1:
            bucket = 'look_down'
        elif progress < split2:
            bucket = 'frontal'
        elif progress < split3:
            bucket = 'look_up'
        elif progress < split4:
            bucket = 'roll_left'
        else:
            bucket = 'roll_right'

        if self.movement_buckets[bucket] >= self.max_per_bucket:
            return False

        self.movement_buckets[bucket] += 1
        return True
    
    def get_processor_name(self) -> str:
        return "VERTICAL_ROLL_PROCESSOR"
    
    def _print_statistics(self, frames: List[FrameQuality]):
        """Override với movement distribution."""
        super()._print_statistics(frames)
        
        logger.info("\n  Phân bố movements:")
        for bucket, count in self.movement_buckets.items():
            logger.info("    %-15s : %d frames", bucket, count)


# ==================== FRAME EXTRACTION SERVICE ====================
class FrameExtractionService:
    """
    Stateless orchestrator cho frame extraction.
    
    Migrated từ Face_Extract_Frames.py - Logic giữ nguyên 100%.
    
    Architecture:
    - Service = orchestrator only
    - Delegates to processor classes (Frontal, Horizontal, Vertical)
    - StorageAdapter: Caller chịu trách nhiệm download/upload
    
    Workflow:
        1. extract_from_video() - Create processor → Extract frames
        2. Processor handles: Quality analysis, face detection, adaptive thresholding, bucketing
    
    Usage:
        from core.adapters import StorageAdapter
        from core.services import FrameExtractionService
        
        storage = StorageAdapter()
        service = FrameExtractionService(storage)
        
        # Extract frames từ video (local path)
        frame_paths, metadata_path = service.extract_from_video(
            video_path=Path("temp/video.mp4"),
            person_name="john",
            video_type="frontal",
            output_dir=Path("temp/frames/frontal"),
            target_frames=100
        )
        
        # Caller upload frames to storage nếu cần
        for frame_path in frame_paths:
            storage.put_file(frame_path, storage_key)
    
    Note:
    - Service xử lý LOCAL files only (temp directories)
    - Batch logic KHÔNG có trong service (thuộc orchestration layer)
    - Không có BatchVideoProcessor (đúng theo Clean Architecture)
    """
    
    def __init__(self, storage: Optional[StorageAdapter] = None):
        """
        Khởi tạo FrameExtractionService.

        Args:
            storage: StorageAdapter instance (optional)
                    Only needed if you plan to upload frames to MinIO
                    For local temp file processing, pass None
        """
        self.storage = storage
    
    def extract_from_video(
        self,
        video_path: Path,
        person_name: str,
        video_type: str,
        output_dir: Path,
        target_frames: int = 100,
    ) -> Tuple[List[Path], Optional[Path]]:
        """
        Extract frames từ video file.
        
        Migrated từ BaseVideoProcessor.extract_frames() - Logic giữ nguyên 100%.
        
        Workflow:
        1. Create appropriate processor (Frontal/Horizontal/Vertical)
        2. Processor analyzes video quality (adaptive thresholding)
        3. Processor detects fast motion và adjusts thresholds
        4. Processor extracts candidate frames (với face detection)
        5. Processor selects best frames (spatial distribution)
        6. Processor saves frames + metadata
        
        Args:
            video_path: Path to video file (local temp)
            person_name: Person name cho naming
            video_type: Video type (frontal/horizontal/vertical)
            output_dir: Output directory cho frames
            target_frames: Target number of frames to extract (default: 100)
            
        Returns:
            (list of frame paths, metadata csv path or None)
            
        Raises:
            ValueError: If video_type invalid or video cannot be opened
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create appropriate processor
        processor = self._create_processor(
            str(video_path),
            str(output_dir),
            person_name,
            video_type
        )
        
        # Extract frames (processor handles all logic)
        frame_paths = processor.extract_frames(target_frames)
        
        # Build metadata path if enabled
        metadata_path = None
        if CONFIG['save_metadata'] and frame_paths:
            video_name = os.path.splitext(os.path.basename(str(video_path)))[0]
            metadata_filename = f"{person_name}_{video_type}_{video_name}_metadata.csv"
            metadata_path = output_dir / metadata_filename
            
            if not metadata_path.exists():
                logger.warning(f"{WARNING} Metadata file not found: {metadata_path}")
                metadata_path = None
        
        return [Path(p) for p in frame_paths], metadata_path
    
    def _create_processor(self, video_path: str, output_dir: str, 
                         person_name: str, video_type: str) -> BaseVideoProcessor:
        """
        Create appropriate processor based on video type.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory
            person_name: Person name
            video_type: Video type (frontal/horizontal/vertical)
            
        Returns:
            Processor instance
            
        Raises:
            ValueError: If video_type unknown
        """
        video_type_lower = video_type.lower()
        
        if video_type_lower == 'frontal':
            return FrontalExpressionProcessor(video_path, output_dir, person_name)
        elif video_type_lower == 'horizontal':
            return HorizontalRotationProcessor(video_path, output_dir, person_name)
        elif video_type_lower == 'vertical':
            return VerticalRollProcessor(video_path, output_dir, person_name)
        else:
            raise ValueError(f"Unknown video type: {video_type}")


__all__ = ["FrameExtractionService"]

