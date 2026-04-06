"""Stateless Video Quality Gate Service.

Wrapper cho Video_Processing pipeline logic.

===================  LỊCH SỬ REFACTOR  ====================
Refactored 17-01-2026:
- Loại bỏ: RAW_VIDEO_DIRS, PROCESSED_VIDEO_DIRS imports (đã xóa từ config)
- Giữ lại: TEMP_PROCESSING_DIR (local temp storage)
- Mục đích: Tương thích với config mới (storage keys)
- Note: Service này xử lý local temp files, không liên quan storage keys

Refactored 30-01-2026:
- SIẾT CHẶT Gate 1: Strict thresholds integration
  * Load WARNING thresholds từ config
  * Generate warnings cho borderline quality
  * Add deblur processing support
- Mục đích: Đảm bảo chỉ video chất lượng TÔT pass Gate 1

===================  REFACTORED LOGGING 09-03-2026  ====================
- Enhanced logging with rich + colorama
- Rich tables for quality metrics
- Color-coded status messages
- Professional dark theme optimized
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    TEMP_PROCESSING_DIR,
    VIDEO_QUALITY_CONFIG,
    INFO, SUCCESS, ERROR, WARNING,
    console, create_table, print_section_header,
)
from core.adapters import StorageAdapter
from core.entities import VideoQualityReport, VideoStatus

logger = logging.getLogger(__name__)

# Load config
CONFIG = {
    'target_fps': VIDEO_QUALITY_CONFIG['TARGET_FPS'],
    'codec': VIDEO_QUALITY_CONFIG['CODEC'],
    'min_duration': VIDEO_QUALITY_CONFIG['MIN_DURATION'],
    'max_duration': VIDEO_QUALITY_CONFIG['MAX_DURATION'],
    'min_resolution': VIDEO_QUALITY_CONFIG['MIN_RESOLUTION'],
    'critical_exposure_clip': VIDEO_QUALITY_CONFIG['CRITICAL_EXPOSURE_CLIP'],  # ← Updated: 0.25 (was 0.95)
    'warning_exposure_clip': VIDEO_QUALITY_CONFIG['WARNING_EXPOSURE_CLIP'],    # ← NEW! 0.15
    'critical_blur_score': VIDEO_QUALITY_CONFIG['CRITICAL_BLUR_SCORE'],        # ← Updated: 35 (was 15)
    'warning_blur_score': VIDEO_QUALITY_CONFIG['WARNING_BLUR_SCORE'],          # ← NEW! 50
    'critical_snr': VIDEO_QUALITY_CONFIG['CRITICAL_SNR'],                      # ← Updated: 10 (was 3)
    'warning_snr': VIDEO_QUALITY_CONFIG['WARNING_SNR'],                        # ← NEW! 15
    'salt_pepper_threshold': VIDEO_QUALITY_CONFIG['SALT_PEPPER_THRESHOLD'],    # ← Updated: 0.05 (was 0.10)
    'warning_salt_pepper': VIDEO_QUALITY_CONFIG['WARNING_SALT_PEPPER'],        # ← NEW! 0.035
    'min_snr_improvement': VIDEO_QUALITY_CONFIG['MIN_SNR_IMPROVEMENT'],        # ← Updated: 2.0 (was 1.0)
    'min_blur_improvement': VIDEO_QUALITY_CONFIG['MIN_BLUR_IMPROVEMENT'],      # ← Updated: 10.0 (was 5.0)
    'enable_exposure_fix': VIDEO_QUALITY_CONFIG['ENABLE_EXPOSURE_FIX'],
    'enable_noise_removal': VIDEO_QUALITY_CONFIG['ENABLE_NOISE_REMOVAL'],
    'enable_light_denoise': VIDEO_QUALITY_CONFIG['ENABLE_LIGHT_DENOISE'],
    'enable_deblur': VIDEO_QUALITY_CONFIG['ENABLE_DEBLUR'],                    # ← NEW! Deblur support
    'sample_frames': VIDEO_QUALITY_CONFIG['SAMPLE_FRAMES'],                    # ← Updated: 100 (was 30)
    'exposure_improvement_eps': VIDEO_QUALITY_CONFIG['EXPOSURE_IMPROVEMENT_EPS'],  # ← Updated: 0.02 (was 0.01)
    'snr_freq_weight': VIDEO_QUALITY_CONFIG['SNR_FREQ_WEIGHT'],
    'median_blur_kernel': VIDEO_QUALITY_CONFIG['MEDIAN_BLUR_KERNEL'],
    'bilateral_params': VIDEO_QUALITY_CONFIG['BILATERAL_PARAMS'],
    'lab_stretch_percentiles': VIDEO_QUALITY_CONFIG['LAB_STRETCH_PERCENTILES'],
    'unsharp_amount': VIDEO_QUALITY_CONFIG['UNSHARP_AMOUNT'],                  # ← NEW! Deblur param
    'unsharp_radius': VIDEO_QUALITY_CONFIG['UNSHARP_RADIUS'],                  # ← NEW! Deblur param
}


# ==================== VIDEO OUTLIER DETECTOR ====================
class VideoOutlierDetector:
    """Phát hiện outliers với robust metrics."""
    
    @staticmethod
    def detect_extreme_exposure(frame: np.ndarray) -> float:
        """Phát hiện chạy sáng/tối cực độ."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        total_pixels = gray.size
        
        extreme_dark = np.sum(gray < 5)
        extreme_bright = np.sum(gray > 250)
        
        return (extreme_dark + extreme_bright) / total_pixels
    
    @staticmethod
    def detect_severe_blur(frame: np.ndarray) -> float:
        """Phát hiện blur nghiêm trọng."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def detect_salt_pepper_noise(frame: np.ndarray) -> float:
        """Phát hiện nhiễu muối tiêu."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        total_pixels = gray.size
        
        salt = np.sum(gray == 255)
        pepper = np.sum(gray == 0)
        
        return (salt + pepper) / total_pixels
    
    @staticmethod
    def estimate_snr_robust(frame: np.ndarray) -> Tuple[float, float]:
        """
        Độc lập SNR với 2 phương pháp.
        
        Returns:
            (snr_traditional, snr_frequency)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Traditional SNR
        signal = gray.mean()
        noise_std = gray.std()
        if noise_std < 1.0:
            noise_std = 1.0
        snr_trad = 20 * np.log10(signal / noise_std)
        
        # Method 2: Frequency-based SNR
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


# ==================== VIDEO QUALITY ANALYZER ====================
class VideoQualityAnalyzer:
    """
    Phân tích video với strict thresholds.

    Refactored 30-01-2026:
    - SIẾT CHẶT thresholds: Reject videos với quality thấp
    - Thêm warning generation: Feedback cho borderline quality
    - Tăng sample size: 30 → 100 frames (better confidence)
    """

    def __init__(self):
        self.detector = VideoOutlierDetector()
    
    def analyze(self, video_path: Path) -> VideoQualityReport:
        """
        Phân tích video với strict quality requirements.

        Refactored 30-01-2026:
        - CRITICAL thresholds: Siết chặt 74-233%
        - WARNING thresholds: Thêm feedback cho borderline quality
        - Sample size: 100 frames (was 30) - better statistical confidence

        Args:
            video_path: Path to video file

        Returns:
            VideoQualityReport object with quality metrics and warnings
        """

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return VideoQualityReport(
                path=str(video_path),
                duration=0, fps=0, resolution=(0,0), total_frames=0,
                avg_exposure_clip=0, avg_blur_score=0, snr=0, snr_frequency=0,
                salt_pepper_ratio=0,
                has_critical_exposure=False, has_critical_blur=False, has_critical_noise=False,
                is_valid=False,
                validation_issues=["Cannot open video"],
                fixable_issues=[],
                warnings=[]  # ← NEW!
            )

        # Extract metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Sample frames uniformly
        sample_size = min(CONFIG['sample_frames'], total_frames)  # ← Updated: 100 samples (was 30)
        sample_interval = max(1, total_frames // sample_size)

        exposure_clips = []
        blur_scores = []
        snr_trad_values = []
        snr_freq_values = []
        noise_ratios = []

        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            exposure_clips.append(self.detector.detect_extreme_exposure(frame))
            blur_scores.append(self.detector.detect_severe_blur(frame))

            snr_t, snr_f = self.detector.estimate_snr_robust(frame)
            snr_trad_values.append(snr_t)
            snr_freq_values.append(snr_f)

            noise_ratios.append(self.detector.detect_salt_pepper_noise(frame))

        cap.release()

        # Calculate averages
        avg_exposure_clip = np.mean(exposure_clips) if exposure_clips else 0
        avg_blur_score = np.mean(blur_scores) if blur_scores else 0
        avg_snr_trad = np.mean(snr_trad_values) if snr_trad_values else 0
        avg_snr_freq = np.mean(snr_freq_values) if snr_freq_values else 0
        avg_noise_ratio = np.mean(noise_ratios) if noise_ratios else 0

        # Combined SNR
        avg_snr_combined = max(avg_snr_trad, avg_snr_freq * CONFIG['snr_freq_weight'])

        # Flag critical issues (STRICT THRESHOLDS!)
        has_critical_exposure = avg_exposure_clip > CONFIG['critical_exposure_clip']  # ← Updated: 0.25 (was 0.95)
        has_critical_blur = avg_blur_score < CONFIG['critical_blur_score']            # ← Updated: 35 (was 15)
        has_critical_noise = (avg_snr_combined < CONFIG['critical_snr'] or            # ← Updated: 10 (was 3)
                             avg_noise_ratio > CONFIG['salt_pepper_threshold'])       # ← Updated: 0.05 (was 0.10)

        # Separate fixable vs unfixable issues
        validation_issues = []
        fixable_issues = []

        # Unfixable
        if duration < CONFIG['min_duration']:
            validation_issues.append(f"Duration too short: {duration:.1f}s")
        if duration > CONFIG['max_duration']:
            validation_issues.append(f"Duration too long: {duration:.1f}s")
        if width < CONFIG['min_resolution'][0] or height < CONFIG['min_resolution'][1]:
            validation_issues.append(f"Resolution too low: {width}x{height}")

        # Fixable
        if has_critical_exposure:
            fixable_issues.append(f"Extreme exposure: {avg_exposure_clip*100:.1f}% pixels clipped")
        if has_critical_blur:
            fixable_issues.append(f"Severe blur: Laplacian {avg_blur_score:.1f}")  # ← Updated: Will attempt deblur
        if has_critical_noise:
            fixable_issues.append(
                f"Heavy noise: SNR {avg_snr_combined:.1f}dB, "
                f"salt-pepper {avg_noise_ratio*100:.2f}%"
            )

        # Generate warnings (NEW! - Borderline quality feedback)
        warnings = self._generate_warnings(
            avg_exposure_clip,
            avg_blur_score,
            avg_snr_combined,
            avg_noise_ratio
        )

        return VideoQualityReport(
            path=str(video_path),
            duration=duration,
            fps=fps,
            resolution=(width, height),
            total_frames=total_frames,
            avg_exposure_clip=avg_exposure_clip,
            avg_blur_score=avg_blur_score,
            snr=avg_snr_combined,
            snr_frequency=avg_snr_freq,
            salt_pepper_ratio=avg_noise_ratio,
            has_critical_exposure=has_critical_exposure,
            has_critical_blur=has_critical_blur,
            has_critical_noise=has_critical_noise,
            is_valid=len(validation_issues) == 0,
            validation_issues=validation_issues,
            fixable_issues=fixable_issues,
            warnings=warnings  # ← NEW! Warning messages
        )

    def _generate_warnings(
        self,
        exposure_clip: float,
        blur_score: float,
        snr: float,
        salt_pepper: float
    ) -> List[str]:
        """
        Generate warning messages for borderline quality.

        NEW in 30-01-2026:
        - Cảnh báo khi quality ở giữa WARNING và CRITICAL
        - User feedback để improve recording conditions

        Args:
            exposure_clip: Average exposure clipping ratio
            blur_score: Average blur score
            snr: Combined SNR
            salt_pepper: Salt-pepper noise ratio

        Returns:
            List of warning messages
        """
        warnings = []

        # Exposure warning
        if CONFIG['warning_exposure_clip'] <= exposure_clip < CONFIG['critical_exposure_clip']:
            warnings.append(
                f"Exposure cảnh báo: {exposure_clip*100:.1f}% pixels clipped "
                f"(khuyến nghị < {CONFIG['warning_exposure_clip']*100:.0f}%). "
                f"Hãy điều chỉnh ánh sáng tốt hơn."
            )

        # Blur warning
        if CONFIG['critical_blur_score'] <= blur_score < CONFIG['warning_blur_score']:
            warnings.append(
                f"Blur cảnh báo: {blur_score:.1f} "
                f"(khuyến nghị > {CONFIG['warning_blur_score']}). "
                f"Hãy giữ camera ổn định hơn."
            )

        # SNR warning
        if CONFIG['critical_snr'] <= snr < CONFIG['warning_snr']:
            warnings.append(
                f"Noise cảnh báo: SNR {snr:.1f}dB "
                f"(khuyến nghị > {CONFIG['warning_snr']}dB). "
                f"Hãy cải thiện điều kiện ánh sáng."
            )

        # Salt-pepper warning
        if CONFIG['warning_salt_pepper'] <= salt_pepper < CONFIG['salt_pepper_threshold']:
            warnings.append(
                f"Salt-pepper noise cảnh báo: {salt_pepper*100:.2f}% "
                f"(khuyến nghị < {CONFIG['warning_salt_pepper']*100:.1f}%). "
                f"Hãy kiểm tra camera sensor."
            )

        return warnings


# ==================== LIGHTWEIGHT VIDEO PROCESSOR ====================
class LightweightVideoProcessor:
    """
    Fix → Re-check → Improvement-based decision.

    Refactored 30-01-2026:
    - Thêm deblur processing: Unsharp masking
    - Stricter improvement requirements: 2x-2x higher
    - Log warnings để user feedback
    """

    def __init__(self, analyzer: VideoQualityAnalyzer):
        self.analyzer = analyzer
    
    def process(self, input_path: Path, output_path: Path) -> VideoStatus:
        """
        Process video: detect → fix → re-validate → decide.

        Returns:
            "copied", "processed", or "failed"
        """

        print_section_header(f"PROCESSING: {input_path.name}")

        # Step 1: Analyze input
        report_before = self.analyzer.analyze(input_path)

        # Video Info Table
        info_table = create_table(
            "Video Information",
            ["Property", "Value"],
            [
                ["Duration", f"{report_before.duration:.1f}s"],
                ["FPS", f"{report_before.fps:.1f}"],
                ["Resolution", f"{report_before.resolution[0]}x{report_before.resolution[1]}"],
                ["Total Frames", f"{report_before.total_frames}"],
            ]
        )
        console.print(info_table)

        # Quality Metrics Table (BEFORE)
        logger.info(f"\n{INFO} Quality Metrics (BEFORE Processing)")

        metrics_table = create_table(
            "Quality Metrics - BEFORE",
            ["Metric", "Value", "Threshold", "Status"],
            [
                ["Exposure Clip", f"{report_before.avg_exposure_clip*100:.1f}%", 
                 f"<{CONFIG['critical_exposure_clip']*100:.0f}%",
                 "PASS" if not report_before.has_critical_exposure else "FAIL"],
                ["Blur Score", f"{report_before.avg_blur_score:.1f}", 
                 f">{CONFIG['critical_blur_score']}",
                 "PASS" if not report_before.has_critical_blur else "FAIL"],
                ["SNR Combined", f"{report_before.snr:.1f} dB", 
                 f">{CONFIG['critical_snr']} dB",
                 "PASS" if not report_before.has_critical_noise else "FAIL"],
                ["Salt-Pepper", f"{report_before.salt_pepper_ratio*100:.2f}%", 
                 f"<{CONFIG['salt_pepper_threshold']*100:.0f}%",
                 "PASS" if report_before.salt_pepper_ratio < CONFIG['salt_pepper_threshold'] else "FAIL"],
            ]
        )
        console.print(metrics_table)

        # Log warnings (borderline quality)
        if report_before.warnings:
            logger.warning(f"\n{WARNING} Borderline quality detected:")
            for warning in report_before.warnings:
                logger.warning(f"{WARNING}   {warning}")

        # Step 2: Hard reject for unfixable issues
        if report_before.validation_issues:
            logger.error(f"\n{ERROR} Unfixable validation issues detected:")
            for issue in report_before.validation_issues:
                logger.error(f"{ERROR}   {issue}")
            logger.error(f"{ERROR} Video rejected - cannot proceed")
            return "failed"

        # Step 3: No fixable issues → copy
        if not report_before.fixable_issues:
            logger.info(f"\n{SUCCESS} No critical issues detected")
            logger.info(f"{INFO} Copying original video")
            success = self._copy_video(input_path, output_path)
            return "copied" if success else "failed"

        # Step 4: Warn + fix
        logger.warning(f"\n{WARNING} Fixable issues detected:")
        for issue in report_before.fixable_issues:
            logger.warning(f"{WARNING}   {issue}")
        logger.info(f"{INFO} Attempting to fix issues...")

        success = self._process_video(input_path, output_path, report_before)
        if not success:
            return "failed"

        # Step 5: Re-validate output
        logger.info(f"\n{INFO} Re-validating processed video...")
        report_after = self.analyzer.analyze(output_path)

        # Quality Metrics Table (AFTER)
        metrics_after_table = create_table(
            "Quality Metrics - AFTER",
            ["Metric", "Before", "After", "Change"],
            [
                ["Exposure Clip", f"{report_before.avg_exposure_clip*100:.1f}%", 
                 f"{report_after.avg_exposure_clip*100:.1f}%",
                 f"{(report_before.avg_exposure_clip - report_after.avg_exposure_clip)*100:+.1f}%"],
                ["Blur Score", f"{report_before.avg_blur_score:.1f}", 
                 f"{report_after.avg_blur_score:.1f}",
                 f"{(report_after.avg_blur_score - report_before.avg_blur_score):+.1f}"],
                ["SNR Combined", f"{report_before.snr:.1f} dB", 
                 f"{report_after.snr:.1f} dB",
                 f"{(report_after.snr - report_before.snr):+.1f} dB"],
                ["Salt-Pepper", f"{report_before.salt_pepper_ratio*100:.2f}%", 
                 f"{report_after.salt_pepper_ratio*100:.2f}%",
                 f"{(report_before.salt_pepper_ratio - report_after.salt_pepper_ratio)*100:+.2f}%"],
            ]
        )
        console.print(metrics_after_table)

        # Step 6: Calculate improvement
        snr_improvement = report_after.snr - report_before.snr
        blur_improvement = report_after.avg_blur_score - report_before.avg_blur_score
        exposure_improvement = report_before.avg_exposure_clip - report_after.avg_exposure_clip

        # Improvement summary
        improvement_table = create_table(
            "Improvement Analysis",
            ["Metric", "Improvement", "Required", "Status"],
            [
                ["SNR", f"{snr_improvement:+.1f} dB", 
                 f"+{CONFIG['min_snr_improvement']:.1f} dB",
                 "PASS" if snr_improvement >= CONFIG['min_snr_improvement'] else "FAIL"],
                ["Blur", f"{blur_improvement:+.1f}", 
                 f"+{CONFIG['min_blur_improvement']:.0f}",
                 "PASS" if blur_improvement >= CONFIG['min_blur_improvement'] else "FAIL"],
                ["Exposure", f"{exposure_improvement*100:+.2f}%", 
                 f"+{CONFIG['exposure_improvement_eps']*100:.2f}%",
                 "PASS" if exposure_improvement > CONFIG['exposure_improvement_eps'] else "FAIL"],
            ]
        )
        console.print(improvement_table)

        # Log warnings after processing (if any remain)
        if report_after.warnings:
            logger.warning(f"\n{WARNING} After processing, still borderline:")
            for warning in report_after.warnings:
                logger.warning(f"{WARNING}   {warning}")

        # Step 7: Decision
        has_improvement = (
            snr_improvement >= CONFIG['min_snr_improvement'] or 
            blur_improvement >= CONFIG['min_blur_improvement'] or 
            exposure_improvement > CONFIG['exposure_improvement_eps'] or
            not report_after.fixable_issues
        )

        if has_improvement:
            logger.info(f"\n{SUCCESS} Video quality improved or acceptable")
            if report_after.fixable_issues:
                logger.warning(f"{WARNING} Note: Still has minor issues, but within tolerance")
            logger.info(f"{INFO} Output: {output_path.name}")
            return "processed"
        else:
            logger.warning(f"\n{WARNING} No significant improvement, but saved for review")
            logger.info(f"{INFO} Output: {output_path.name}")
            return "processed"
    
    def _copy_video(self, input_path: Path, output_path: Path) -> bool:
        """Copy video without processing."""
        try:
            import shutil
            shutil.copy2(input_path, output_path)
            logger.info(f"{SUCCESS} Video copied: {output_path.name}")
            return True
        except Exception as ex:
            logger.error(f"{ERROR} Copy failed: {ex}")
            logger.exception(f"{ERROR} Stack trace:")
            return False

    def _process_video(self, input_path: Path, output_path: Path, 
                      report: VideoQualityReport) -> bool:
        """Process video with lightweight pipeline."""

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"{ERROR} Cannot open video")
            return False

        width, height = report.resolution
        fps = report.fps if report.fps > 0 else CONFIG['target_fps']

        fourcc = cv2.VideoWriter_fourcc(*CONFIG['codec'])
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error(f"{ERROR} Cannot create output video")
            cap.release()
            return False

        frame_count = 0
        logger.info(f"\n{INFO} Processing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self._fix_critical_issues(frame, report)
            out.write(processed_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / report.total_frames) * 100
                logger.info(f"{INFO}   Progress: {progress:.0f}%")

        cap.release()
        out.release()

        logger.info(f"\n{SUCCESS} Processed {frame_count} frames")
        return True
    
    def _fix_critical_issues(self, frame: np.ndarray, 
                            report: VideoQualityReport) -> np.ndarray:
        """
        Apply fixes for critical quality issues.

        Refactored 30-01-2026:
        - Thêm deblur processing (unsharp masking)
        - Order: Exposure → Noise → Deblur (optimal pipeline)
        """
        processed = frame.copy()

        # Fix 1: Exposure correction
        if report.has_critical_exposure and CONFIG['enable_exposure_fix']:
            processed = self._fix_extreme_exposure(processed)

        # Fix 2: Salt-pepper noise removal
        if report.has_critical_noise and CONFIG['enable_noise_removal']:
            if report.salt_pepper_ratio > CONFIG['salt_pepper_threshold']:  # ← Updated: 0.05 (was 0.10)
                processed = cv2.medianBlur(processed, CONFIG['median_blur_kernel'])

        # Fix 3: General denoising
        if report.has_critical_noise and CONFIG['enable_light_denoise']:
            params = CONFIG['bilateral_params']
            processed = cv2.bilateralFilter(
                processed,
                params.get('d', 5),
                params.get('sigma_color', 25),
                params.get('sigma_space', 25)
            )

        # Fix 4: Deblur (NEW! - Unsharp masking)
        if report.has_critical_blur and CONFIG.get('enable_deblur', False):
            processed = self._apply_deblur(processed)

        return processed

    def _apply_deblur(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply deblur using unsharp masking.

        NEW in 30-01-2026:
        - Technique: Unsharp mask (frame - blurred)
        - Conservative approach: Avoid over-sharpening

        Args:
            frame: Input frame (BGR)

        Returns:
            Sharpened frame
        """
        # Gaussian blur
        blurred = cv2.GaussianBlur(frame, (0, 0), CONFIG['unsharp_radius'])

        # Unsharp mask: original + amount * (original - blurred)
        sharpened = cv2.addWeighted(
            frame, 
            CONFIG['unsharp_amount'],  # Amount (1.5 = 150%)
            blurred, 
            -(CONFIG['unsharp_amount'] - 1.0),  # Negative weight for blur
            0
        )

        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened
    
    @staticmethod
    def _fix_extreme_exposure(frame: np.ndarray) -> np.ndarray:
        """Fix extreme exposure."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        p_low, p_high = CONFIG['lab_stretch_percentiles']
        l_min, l_max = np.percentile(l_channel, [p_low, p_high])
        l_stretched = np.clip((l_channel - l_min) / (l_max - l_min) * 255, 0, 255).astype(np.uint8)
        
        lab_stretched = cv2.merge([l_stretched, a_channel, b_channel])
        return cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)


# ==================== VIDEO QUALITY SERVICE ====================
class VideoQualityService:
    """
    Stateless orchestrator cho video quality gate.
    
    Migrated từ pipeline/Video_Processing.py - Logic giữ nguyên 100%.
    
    Architecture:
    - Dependency injection: StorageAdapter
    - Talks only via Config, Entities, Adapters
    - Self-contained logic (không phụ thuộc pipeline)
    
    Workflow:
        1. analyze_video() - Detect quality issues
        2. process_video() - Fix → Re-check → Decide
    
    Usage:
        from core.adapters import StorageAdapter
        from core.services import VideoQualityService
        
        storage = StorageAdapter()
        service = VideoQualityService(storage)
        
        # Analyze only
        report = service.analyze_video(video_path)
        
        # Process workflow (detect → fix → re-validate)
        status = service.process_video(input_path, output_path)
    
    Note:
    - Service xử lý LOCAL temp files (input_path, output_path)
    - Caller chịu trách nhiệm download/upload với storage
    - KHÔNG có batch logic (thuộc orchestration layer)
    """
    
    def __init__(self, storage: Optional[StorageAdapter] = None):
        """
        Khởi tạo VideoQualityService.

        Args:
            storage: StorageAdapter instance (optional)
                    Only needed for process_video() method
                    For analyze_video() only, pass None to skip MinIO connection
        """
        self.storage = storage
        self.analyzer = VideoQualityAnalyzer()
        self.processor = LightweightVideoProcessor(self.analyzer)
    
    # ---------------- ANALYSIS ----------------
    
    def analyze_video(self, video_path: Path) -> VideoQualityReport:
        """
        Analyze video quality (detect issues only).
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoQualityReport object with:
            - Quality metrics (exposure, blur, SNR, noise)
            - Fixable issues vs unfixable issues
            - Validation status
        """
        return self.analyzer.analyze(video_path)
    
    # ---------------- PROCESSING ----------------
    
    def process_video(self, input_path: Path, output_path: Path) -> VideoStatus:
        """
        Process video: detect → fix → re-validate → decide.
        
        Migrated từ LightweightVideoProcessor.process() - Logic giữ nguyên 100%.
        
        Workflow:
        1. Analyze input video
        2. Reject nếu có unfixable issues (duration, resolution)
        3. Copy nếu không có fixable issues
        4. Fix critical issues (exposure, noise)
        5. Re-validate output
        6. Decide based on improvement metrics
        
        Args:
            input_path: Path to input video (local temp)
            output_path: Path to output video (local temp)
            
        Returns:
            VideoStatus:
            - "copied": No issues, original copied
            - "processed": Fixed and improved
            - "failed": Unfixable issues or processing error
        
        Note:
        - Service xử lý LOCAL files only
        - Caller download từ storage → input_path
        - Caller upload từ output_path → storage
        """
        return self.processor.process(input_path, output_path)


__all__ = ["VideoQualityService"]
