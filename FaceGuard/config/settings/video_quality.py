"""
Video quality configuration - Balanced for production.

===================  LỊCH SỬ REFACTOR  ====================
Refactored 16-01-2026:
- Initial version với relaxed thresholds

Refactored 30-01-2026:
- SIẾT CHẶT Gate 1: Tighten thresholds 74-233%

Refactored 04-03-2026:
- NỚI LỎNG Gate 1: Relax ~30-40% (TOO LOOSE!)

Refactored 08-03-2026:
- BALANCED Gate 1: Re-tighten to production-ready levels
  * CRITICAL_EXPOSURE_CLIP: 0.35 → 0.15 (57% stricter)
  * WARNING_EXPOSURE_CLIP: 0.25 → 0.10
  * CRITICAL_BLUR_SCORE: 25 → 40 (60% stricter)
  * WARNING_BLUR_SCORE: 35 → 60
  * CRITICAL_SNR: 7 → 10 dB (43% stricter)
  * WARNING_SNR: 10 → 15 dB
  * SALT_PEPPER_THRESHOLD: 0.07 → 0.04 (43% stricter)
  * WARNING_SALT_PEPPER: 0.05 → 0.03
- Mục đích: Balance quality vs collection ease
- Rationale: Based on user test (80%+ in good conditions)
- Standard: Indoor face recognition industry best practices

Architecture:
Gate 1 (Video) → Gate 2 (Frame) → Gate 3 (Embedding) → Gate 4 (Sanitizer)
Kim tự tháp: Càng về sau càng strict
"""
from .path import TEMP_DIR

# ==================== LOCAL PATHS (Temp only) ====================
# Removed auto-creation: These dirs are created on-demand by services
TEMP_PROCESSING_DIR = TEMP_DIR / "video_processing"  # Used by VideoQualityService (on-demand)

# ==================== CONFIG ====================

VIDEO_QUALITY_CONFIG = {
    # ============ BASIC SETTINGS ============
    "TARGET_FPS": 30,
    "CODEC": "mp4v",
    "MIN_DURATION": 10.0,
    "MAX_DURATION": 20.0,
    "MIN_RESOLUTION": (480, 360),

    # ============ EXPOSURE THRESHOLDS ============
    # Exposure clip: % of pixels bị cháy (>250) hoặc tối (<5)
    # Industry standard: <10-15% for good quality

    "CRITICAL_EXPOSURE_CLIP": 0.15,    # ✅ BALANCED: 15% max (was 35%)
    "WARNING_EXPOSURE_CLIP": 0.10,     # ✅ BALANCED: 10% warning (was 25%)

    # Rationale: 15% allows some over/under exposure but not extreme
    # User test: Good conditions typically have <5% clipping

    # ============ BLUR THRESHOLDS ============
    # Blur score: Laplacian variance (higher = sharper)
    # Sharp video: >100, Acceptable: 40-80, Blurry: <40

    "CRITICAL_BLUR_SCORE": 40,         # ✅ BALANCED: 40 min (was 25)
    "WARNING_BLUR_SCORE": 60,          # ✅ BALANCED: 60 recommended (was 35)

    # Rationale: 40 = acceptable sharpness, 60 = good quality
    # User test: Stable camera typically scores 80-120

    # ============ SNR THRESHOLDS ============
    # SNR: Signal-to-Noise Ratio in dB (higher = less noise)
    # Good: >15dB, Acceptable: 10-15dB, Poor: <10dB

    "CRITICAL_SNR": 10,                # ✅ BALANCED: 10dB min (was 7)
    "WARNING_SNR": 15,                 # ✅ BALANCED: 15dB recommended (was 10)

    # Rationale: 10dB = minimum acceptable, 15dB = good quality
    # User test: Good lighting typically achieves 18-25dB

    # ============ SALT-PEPPER NOISE ============
    # Salt-pepper: % of pixels that are pure white/black noise
    # Good: <2%, Acceptable: 2-4%, Poor: >4%

    "SALT_PEPPER_THRESHOLD": 0.04,     # ✅ BALANCED: 4% max (was 7%)
    "WARNING_SALT_PEPPER": 0.03,       # ✅ BALANCED: 3% warning (was 5%)

    # Rationale: 4% = borderline acceptable, 3% = good quality
    # User test: Clean camera sensor typically <2%

    # ============ IMPROVEMENT THRESHOLDS ============
    # Re-validation requirements (after processing)

    "MIN_SNR_IMPROVEMENT": 2.0,        # Need +2dB improvement
    "MIN_BLUR_IMPROVEMENT": 10.0,      # Need +10 Laplacian improvement
    "EXPOSURE_IMPROVEMENT_EPS": 0.02,  # Need 2% exposure reduction

    # ============ PROCESSING OPTIONS ============
    "ENABLE_EXPOSURE_FIX": True,       # Fix over/under exposure
    "ENABLE_NOISE_REMOVAL": True,      # Remove salt-pepper noise
    "ENABLE_LIGHT_DENOISE": True,      # Bilateral filtering
    "ENABLE_DEBLUR": True,             # Unsharp masking

    # ============ SAMPLING ============
    "SAMPLE_FRAMES": 100,              # Sample 100 frames (was 30)
                                       # Better statistical confidence

    # ============ PROCESSING PARAMETERS ============
    "SNR_FREQ_WEIGHT": 0.6,            # Weight for frequency-based SNR
    "MEDIAN_BLUR_KERNEL": 3,           # Salt-pepper removal kernel
    "BILATERAL_PARAMS": {              # Denoising parameters
        "d": 5, 
        "sigma_color": 25, 
        "sigma_space": 25
    },
    "LAB_STRETCH_PERCENTILES": (2, 98),  # Exposure correction range

    # Deblur parameters (Unsharp masking)
    "UNSHARP_AMOUNT": 1.5,             # Sharpening strength
    "UNSHARP_RADIUS": 3,               # Sharpening radius
}

__all__ = [
    "TEMP_PROCESSING_DIR",
    "VIDEO_QUALITY_CONFIG",
]
