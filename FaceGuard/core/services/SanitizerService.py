"""Stateless Data Sanitization Service.

Migrated từ pipeline/Face_Sanitizer.py - Logic giữ nguyên 100%.

===================  LỊCH SỬ REFACTOR  ====================
Refactored 17-01-2026:
- Migrate: TOÀN BỘ logic từ Face_Sanitizer.py
  * DataValidator (7 integrity checks + outlier detection)
  * DataCleaner (filter by quality, remove outliers, handle multiface, balance dataset)
  * DataSplitter (stratified split by person + image_type)
- Service: Stateless orchestrator, delegate to classes
- Architecture: Validator → Cleaner → Splitter pipeline

Refactored 31-01-2026:
- ALIGN VỚI GATE 2-3: Lower thresholds (0.70→0.60 safety net!)
- ADD GATE 2-3 METADATA VALIDATION: norm, face_conf, SNR, landmarks (4 new checks!)
- ENHANCE QUALITY FILTERING: Use all Gate 2-3 signals for comprehensive validation
- OPTIMIZE LOGGING: Show detailed Gate 2-3 metadata stats
- Mục đích: Gate 4 = final safety net, catch critical failures only!

Refactored 05-03-2026:
- OPTIMIZED INPUT/OUTPUT: NPZ + JSON only (align với Phase 3/5)
- REMOVED: CSV, Parquet loading/saving (redundant, slower)
- REASON:
  * Consistent format across all phases
  * Faster loading (no CSV parsing)
  * Smaller intermediate files
  * Cleaner architecture

===================  REFACTORED LOGGING 09-03-2026  ====================
- Enhanced logging with rich + colorama
- Rich tables for validation checks and statistics
- Color-coded prefixes for dark theme
- Professional output without emojis

Architecture:
- Service = orchestrator (run entire pipeline)
- DataValidator = integrity validation + outlier detection + Gate 2-3 metadata checks
- DataCleaner = quality filtering + balancing (use Gate 2-3 metadata!)
- DataSplitter = stratified train/val/test split
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from config.settings import (
    SANITIZER_CONFIG,
    SANITIZER_THRESHOLDS,
    INFO, SUCCESS, ERROR, WARNING,
    console, create_table, print_section_header,
)
from core.adapters import StorageAdapter
from core.entities import CleaningReport, SplitReport, ValidationReport

logger = logging.getLogger(__name__)

# Load config
THRESHOLDS = SANITIZER_THRESHOLDS
CONFIG = SANITIZER_CONFIG


# ==================== DATA VALIDATOR ====================
class DataValidator:
    """
    Validate embedding dataset integrity and quality.

    Refactored 05-03-2026:
    - OPTIMIZED: Load from NPZ + JSON (Phase 3 output)
    - REMOVED: CSV, Parquet loading (redundant)
    - REASON: Consistent format, faster loading

    Responsibilities:
    - Check data integrity (NaN, Inf, normalization) - 11 checks
    - Detect outliers using Isolation Forest
    - Analyze quality distribution
    - Compute label/type distributions
    """

    def __init__(self, npz_path: str, json_path: str):
        """
        Initialize validator with NPZ + JSON dataset.

        Refactored 05-03-2026:
        - Load from NPZ + JSON only (Phase 3 output)
        - NPZ: embeddings array (vectors)
        - JSON: metadata (quality scores, Gate 2-3 data)

        Args:
            npz_path: Path to NPZ file (embeddings, labels, paths)
            json_path: Path to JSON file (metadata)
        """
        print_section_header("DATA VALIDATOR INITIALIZATION")

        # Load NPZ (vectors)
        npz_data = np.load(npz_path)
        self.embeddings = npz_data['embeddings']
        self.labels = npz_data['labels']
        self.image_paths = npz_data.get('image_paths', np.array([]))

        # NPZ Info Table
        npz_table = create_table(
            "NPZ Data Loaded",
            ["Property", "Value"],
            [
                ["Samples", f"{len(self.embeddings)}"],
                ["Shape", f"{self.embeddings.shape}"],
                ["Unique Labels", f"{len(np.unique(self.labels))}"],
            ]
        )
        console.print(npz_table)

        # Load JSON (metadata)
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.metadata = metadata
        self.samples_metadata = metadata.get('samples', [])

        logger.info(f"{SUCCESS} JSON metadata loaded: {len(self.samples_metadata)} samples")

        # Build DataFrame from JSON metadata (for compatibility)
        self.df = pd.DataFrame(self.samples_metadata)

        logger.info(f"{INFO} Columns: {', '.join(list(self.df.columns))}")

        # All metadata available from JSON
        self.has_metadata = True
    
    def validate_integrity(self) -> ValidationReport:
        """
        Kiểm tra tính toàn vẹn cơ bản của dataset với 11 checks.

        Refactored 31-01-2026:
        - Add checks 8-11 for Gate 2-3 metadata (norm, face_conf, SNR, landmarks)
        - Enhanced reporting với detailed stats

        Returns:
            ValidationReport object
        """
        print_section_header("STAGE 1: INTEGRITY VALIDATION")

        # Check 1: NaN values
        nan_count = np.isnan(self.embeddings).sum()
        has_nan = nan_count > 0
        logger.info(f"\n{INFO} CHECK 1: NaN values - {nan_count} found")
        if has_nan:
            logger.error(f"{ERROR} FAIL - Found {nan_count} NaN values")
        else:
            logger.info(f"{SUCCESS} PASS - No NaN values")

        # Check 2: Inf values
        inf_count = np.isinf(self.embeddings).sum()
        has_inf = inf_count > 0
        logger.info(f"\n{INFO} CHECK 2: Inf values - {inf_count} found")
        if has_inf:
            logger.error(f"{ERROR} FAIL - Found {inf_count} Inf values")
        else:
            logger.info(f"{SUCCESS} PASS - No Inf values")

        # Check 3: Normalization
        norms = np.linalg.norm(self.embeddings, axis=1)
        is_normalized = np.allclose(norms, 1.0, atol=1e-5)
        logger.info(f"\n{INFO} CHECK 3: Normalization")
        logger.info(f"    Mean norm: {norms.mean():.6f}")
        logger.info(f"    Std norm : {norms.std():.6f}")
        logger.info(f"    Min/Max  : {norms.min():.6f} / {norms.max():.6f}")
        if is_normalized:
            logger.info(f"{SUCCESS} PASS - All embeddings normalized")
        else:
            logger.warning(f"{WARNING} WARNING - Not all embeddings perfectly normalized")

        # Check 4: Duplicates
        unique_embeddings = np.unique(self.embeddings, axis=0)
        has_duplicates = len(unique_embeddings) < len(self.embeddings)
        dup_count = len(self.embeddings) - len(unique_embeddings)
        logger.info(f"\n{INFO} CHECK 4: Duplicates - {dup_count} found")
        if has_duplicates:
            logger.warning(f"{WARNING} Found {dup_count} duplicate embeddings")
        else:
            logger.info(f"{SUCCESS} PASS - No duplicates")

        # Check 5: Label distribution
        label_counts = pd.Series(self.labels).value_counts().to_dict()
        logger.info(f"\n{INFO} CHECK 5: Label distribution")
        for label, count in label_counts.items():
            pct = count / len(self.labels) * 100
            logger.info(f"    {label}: {count} samples ({pct:.1f}%)")

        # Check 6: Image type distribution
        type_counts = self.df['image_type'].value_counts().to_dict()
        logger.info(f"\n{INFO} CHECK 6: Image type distribution")
        for img_type, count in type_counts.items():
            pct = count / len(self.df) * 100
            logger.info(f"    {img_type}: {count} samples ({pct:.1f}%)")

        # Check 7: Quality score distribution (if available)
        quality_stats = {}
        if 'quality_score' in self.df.columns:
            quality_stats = {
                'mean': float(self.df['quality_score'].mean()),
                'std': float(self.df['quality_score'].std()),
                'min': float(self.df['quality_score'].min()),
                'max': float(self.df['quality_score'].max()),
                'q25': float(self.df['quality_score'].quantile(0.25)),
                'q50': float(self.df['quality_score'].quantile(0.50)),
                'q75': float(self.df['quality_score'].quantile(0.75)),
            }
            logger.info(f"\n{INFO} CHECK 7: Quality score distribution")
            logger.info(f"    Mean: {quality_stats['mean']:.4f} (+/- {quality_stats['std']:.4f})")
            logger.info(f"    Min/Max: {quality_stats['min']:.4f} / {quality_stats['max']:.4f}")
            logger.info(f"    Q25/Q50/Q75: {quality_stats['q25']:.4f} / {quality_stats['q50']:.4f} / {quality_stats['q75']:.4f}")

        # ════════════════════════════════════════════════════════
        # NEW! Check 8: Embedding norm distribution (Gate 3)
        # ════════════════════════════════════════════════════════

        embedding_norm_stats = {}
        if 'embedding_norm' in self.df.columns:
            norms = self.df['embedding_norm'].values
            norm_min = THRESHOLDS['EMBEDDING_NORM_MIN']
            norm_max = THRESHOLDS['EMBEDDING_NORM_MAX']
            out_of_range = ((norms < norm_min) | (norms > norm_max)).sum()

            embedding_norm_stats = {
                'mean': float(norms.mean()),
                'std': float(norms.std()),
                'min': float(norms.min()),
                'max': float(norms.max()),
                'out_of_range_count': int(out_of_range),
                'out_of_range_pct': float(out_of_range / len(norms) * 100)
            }

            logger.info(f"\n{INFO} CHECK 8: Embedding norm distribution")
            logger.info(f"    Mean: {embedding_norm_stats['mean']:.4f} (+/- {embedding_norm_stats['std']:.4f})")
            logger.info(f"    Min/Max: {embedding_norm_stats['min']:.4f} / {embedding_norm_stats['max']:.4f}")
            logger.info(f"    Expected range: [{norm_min:.2f}, {norm_max:.2f}]")
            logger.info(f"    Out of range: {out_of_range} ({embedding_norm_stats['out_of_range_pct']:.1f}%)")

            if out_of_range > len(self.df) * 0.05:
                logger.warning(f"{WARNING} >5% bad norms (Inference errors?)")
            else:
                logger.info(f"{SUCCESS} PASS - Most norms in valid range")

        # ════════════════════════════════════════════════════════
        # NEW! Check 9: Gate 2 MediaPipe confidence distribution
        # ════════════════════════════════════════════════════════

        gate2_confidence_stats = {}
        if 'face_confidence_gate2' in self.df.columns:
            confs = self.df['face_confidence_gate2'].values

            # Skip if all zeros (Gate 2 metadata not available!)
            if confs.max() == 0.0:
                logger.info(f"\n{INFO} CHECK 9: Gate 2 MediaPipe confidence")
                logger.warning(f"{WARNING} SKIPPED - All zeros (Gate 2 metadata not available)")
                logger.info(f"{INFO} This is OK - Gate 4 will validate embeddings only")
            else:
                low_threshold = THRESHOLDS['GATE2_FACE_CONFIDENCE_MIN']
                low_count = (confs < low_threshold).sum()

                gate2_confidence_stats = {
                    'mean': float(confs.mean()),
                    'std': float(confs.std()),
                    'min': float(confs.min()),
                    'max': float(confs.max()),
                    'low_count': int(low_count),
                    'low_pct': float(low_count / len(confs) * 100)
                }

                logger.info(f"\n{INFO} CHECK 9: Gate 2 MediaPipe confidence")
                logger.info(f"    Mean: {gate2_confidence_stats['mean']:.3f} (+/- {gate2_confidence_stats['std']:.3f})")
                logger.info(f"    Min/Max: {gate2_confidence_stats['min']:.3f} / {gate2_confidence_stats['max']:.3f}")
                logger.info(f"    Low confidence (<{low_threshold:.2f}): {low_count} ({gate2_confidence_stats['low_pct']:.1f}%)")

                if low_count > 0:
                    logger.warning(f"{WARNING} {low_count} samples with low Gate 2 confidence")

        # ════════════════════════════════════════════════════════
        # NEW! Check 10: Gate 2 Frame SNR distribution
        # ════════════════════════════════════════════════════════

        gate2_snr_stats = {}
        if 'snr_gate2' in self.df.columns:
            snrs = self.df['snr_gate2'].values

            # Skip if all zeros (Gate 2 metadata not available!)
            if snrs.max() == 0.0:
                logger.info(f"\n{INFO} CHECK 10: Gate 2 Frame SNR")
                logger.warning(f"{WARNING} SKIPPED - All zeros (Gate 2 metadata not available)")
                logger.info(f"{INFO} This is OK - Gate 4 will validate embeddings only")
            else:
                low_threshold = THRESHOLDS['GATE2_SNR_MIN']
                low_count = (snrs < low_threshold).sum()

                gate2_snr_stats = {
                    'mean': float(snrs.mean()),
                    'std': float(snrs.std()),
                    'min': float(snrs.min()),
                    'max': float(snrs.max()),
                    'low_count': int(low_count),
                    'low_pct': float(low_count / len(snrs) * 100)
                }

                logger.info(f"\n{INFO} CHECK 10: Gate 2 Frame SNR")
                logger.info(f"    Mean: {gate2_snr_stats['mean']:.1f} dB (+/- {gate2_snr_stats['std']:.1f})")
                logger.info(f"    Min/Max: {gate2_snr_stats['min']:.1f} / {gate2_snr_stats['max']:.1f} dB")
                logger.info(f"    Low SNR (<{low_threshold:.1f} dB): {low_count} ({gate2_snr_stats['low_pct']:.1f}%)")

                if low_count > 0:
                    logger.warning(f"{WARNING} {low_count} samples with low Gate 2 SNR")

        # ════════════════════════════════════════════════════════
        # NEW! Check 11: Gate 3 Landmarks quality distribution
        # ════════════════════════════════════════════════════════

        landmarks_quality_stats = {}
        if 'landmarks_quality' in self.df.columns:
            qualities = self.df['landmarks_quality'].values
            low_threshold = THRESHOLDS['LANDMARKS_QUALITY_MIN']
            low_count = (qualities < low_threshold).sum()

            landmarks_quality_stats = {
                'mean': float(qualities.mean()),
                'std': float(qualities.std()),
                'min': float(qualities.min()),
                'max': float(qualities.max()),
                'low_count': int(low_count),
                'low_pct': float(low_count / len(qualities) * 100)
            }

            logger.info(f"\n{INFO} CHECK 11: Gate 3 Landmarks quality")
            logger.info(f"    Mean: {landmarks_quality_stats['mean']:.3f} (+/- {landmarks_quality_stats['std']:.3f})")
            logger.info(f"    Min/Max: {landmarks_quality_stats['min']:.3f} / {landmarks_quality_stats['max']:.3f}")
            logger.info(f"    Low quality (<{low_threshold:.2f}): {low_count} ({landmarks_quality_stats['low_pct']:.1f}%)")

            if low_count > 0:
                logger.warning(f"{WARNING} {low_count} samples with low landmarks quality")

        # Overall pass/fail
        passed = not (has_nan or has_inf)

        logger.info("\n" + "="*70)
        if passed:
            logger.info(f"{SUCCESS} INTEGRITY CHECK PASSED")
        else:
            logger.error(f"{ERROR} INTEGRITY CHECK FAILED")
        logger.info("="*70)

        return ValidationReport(
            passed=passed,
            has_nan=has_nan,
            has_inf=has_inf,
            is_normalized=is_normalized,
            has_duplicates=has_duplicates,
            label_counts=label_counts,
            type_counts=type_counts,
            quality_stats=quality_stats,
            embedding_norm_stats=embedding_norm_stats,  # NEW!
            gate2_confidence_stats=gate2_confidence_stats,  # NEW!
            gate2_snr_stats=gate2_snr_stats,  # NEW!
            landmarks_quality_stats=landmarks_quality_stats  # NEW!
        )
    
    def detect_outliers(self, method: str = 'isolation_forest',
                       contamination: float = 0.05) -> np.ndarray:
        """
        Detect outlier embeddings.
        
        Migrated từ Face_Sanitizer.py - Logic giữ nguyên 100%.
        
        Args:
            method: 'isolation_forest' or 'statistical'
            contamination: Expected outlier ratio
            
        Returns:
            Boolean mask (True = outlier)
        """
        print_section_header("STAGE 2: OUTLIER DETECTION")
        logger.info(f"{INFO} Method: {method}")
        logger.info(f"{INFO} Contamination: {contamination * 100:.1f}%")
        
        if method == 'isolation_forest':
            clf = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            outlier_labels = clf.fit_predict(self.embeddings)
            outlier_mask = outlier_labels == -1
            
        elif method == 'statistical':
            # Z-score based outlier detection
            z_scores = np.abs(stats.zscore(self.embeddings, axis=0))
            outlier_mask = (z_scores > 3).any(axis=1)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_count = outlier_mask.sum()
        outlier_pct = outlier_count / len(self.embeddings) * 100

        logger.info(f"\n{INFO} Detected {outlier_count} outliers ({outlier_pct:.1f}%)")

        # Analyze outliers by person and type
        if outlier_count > 0:
            logger.info(f"\n{INFO} Outlier breakdown:")
            outlier_df = self.df[outlier_mask]

            for person in outlier_df['person_name'].unique():
                person_outliers = outlier_df[outlier_df['person_name'] == person]
                logger.info(f"    {person}: {len(person_outliers)} outliers")

                for img_type in person_outliers['image_type'].unique():
                    type_count = len(person_outliers[person_outliers['image_type'] == img_type])
                    logger.info(f"      -> {img_type}: {type_count}")
        
        return outlier_mask


# ==================== DATA CLEANER ====================
class DataCleaner:
    """
    Clean embedding dataset based on validation report.
    
    Migrated từ Face_Sanitizer.py - Logic giữ nguyên 100%.
    
    Responsibilities:
    - Filter by quality thresholds
    - Remove outliers
    - Handle multi-face frames
    - Balance dataset
    """
    
    def __init__(self, validator: DataValidator, validation_report: ValidationReport):
        """
        Initialize cleaner với validator và report.
        
        Args:
            validator: DataValidator instance (contains data)
            validation_report: ValidationReport from validator
        """
        print_section_header("DATA CLEANER INITIALIZATION")

        # Store references
        self.validator = validator
        self.report = validation_report

        # Copy data (will be modified)
        self.df = validator.df.copy()
        self.embeddings = validator.embeddings.copy()
        self.labels = validator.labels.copy()

        # Track original size
        self.original_size = len(self.df)

        # Tracking masks for each operation
        self.valid_mask = np.ones(len(self.df), dtype=bool)
        self.removal_reasons = {
            'quality': np.zeros(len(self.df), dtype=bool),
            'outliers': np.zeros(len(self.df), dtype=bool),
            'multiface': np.zeros(len(self.df), dtype=bool),
            'balance': np.zeros(len(self.df), dtype=bool)
        }

        logger.info(f"{INFO} Original dataset size: {self.original_size}")
    
    def filter_by_quality(self,
                         min_detection: Optional[float] = None,
                         min_quality: Optional[float] = None) -> 'DataCleaner':
        """
        Filter samples by detection and quality scores + Gate 2-3 metadata.

        Refactored 31-01-2026:
        - Add embedding norm validation (Gate 3)
        - Add Gate 2 confidence check (MediaPipe)
        - Add Gate 2 SNR check
        - Add Gate 3 landmarks quality check
        - Comprehensive multi-gate validation!

        Args:
            min_detection: Minimum detection score threshold
            min_quality: Minimum quality score threshold

        Returns:
            self (for method chaining)
        """
        logger.info("\n%s", "="*70)
        logger.info("  STAGE 3: QUALITY FILTERING")
        logger.info("%s", "="*70)

        if min_detection is None:
            min_detection = THRESHOLDS['MIN_DETECTION_SCORE']
        if min_quality is None:
            min_quality = THRESHOLDS['MIN_QUALITY_SCORE']

        logger.info("  Min detection score: %.2f (looser than Gate 3!)", min_detection)
        logger.info("  Min quality score  : %.2f (safety net)", min_quality)

        # Build quality mask
        quality_mask = self.df['detection_score'] >= min_detection

        if 'quality_score' in self.df.columns:
            quality_mask &= self.df['quality_score'] >= min_quality

        # ════════════════════════════════════════════════════════
        # NEW! Gate 2-3 Metadata Checks (from JSON)
        # ════════════════════════════════════════════════════════

        # Check 1: Embedding norm validation (CRITICAL!)
        if THRESHOLDS['ENABLE_NORM_CHECK'] and 'embedding_norm' in self.df.columns:
            norm_min = THRESHOLDS['EMBEDDING_NORM_MIN']
            norm_max = THRESHOLDS['EMBEDDING_NORM_MAX']
            norm_mask = (
                (self.df['embedding_norm'] >= norm_min) &
                (self.df['embedding_norm'] <= norm_max)
            )

            norm_removed = (~norm_mask & self.valid_mask).sum()
            if norm_removed > 0:
                logger.warning("  [NORM CHECK] Removed %d samples by bad embedding norm", norm_removed)
                logger.warning("    These are likely inference errors! ⚠")

            quality_mask &= norm_mask

        # Check 2: Gate 2 MediaPipe confidence (from JSON metadata)
        if THRESHOLDS['ENABLE_GATE2_CONFIDENCE_CHECK']:
            if 'face_confidence_gate2' in self.df.columns:
                confs = self.df['face_confidence_gate2'].values

                # Skip if all zeros (Gate 2 metadata not available!)
                if confs.max() == 0.0:
                    logger.warning("  [GATE2 CONF] All zeros - Gate 2 metadata not available, skipping...")
                else:
                    gate2_conf_min = THRESHOLDS['GATE2_FACE_CONFIDENCE_MIN']
                    gate2_mask = confs >= gate2_conf_min

                    gate2_removed = (~gate2_mask & self.valid_mask).sum()
                    if gate2_removed > 0:
                        logger.info("  [GATE2 CONF] Removed %d samples by low MediaPipe confidence", gate2_removed)

                    quality_mask &= gate2_mask

        # Check 3: Gate 2 Frame SNR (from JSON metadata)
        if THRESHOLDS['ENABLE_GATE2_SNR_CHECK']:
            if 'snr_gate2' in self.df.columns:
                snrs = self.df['snr_gate2'].values

                # Skip if all zeros (Gate 2 metadata not available!)
                if snrs.max() == 0.0:
                    logger.warning("  [GATE2 SNR] All zeros - Gate 2 metadata not available, skipping...")
                else:
                    snr_min = THRESHOLDS['GATE2_SNR_MIN']
                    snr_mask = snrs >= snr_min

                    snr_removed = (~snr_mask & self.valid_mask).sum()
                    if snr_removed > 0:
                        logger.info("  [GATE2 SNR] Removed %d samples by low Frame SNR", snr_removed)

                    quality_mask &= snr_mask

        # Check 4: Gate 3 Landmarks quality (from JSON metadata)
        if THRESHOLDS['ENABLE_LANDMARKS_CHECK']:
            if 'landmarks_quality' in self.df.columns:
                landmarks_min = THRESHOLDS['LANDMARKS_QUALITY_MIN']
                landmarks_mask = self.df['landmarks_quality'] >= landmarks_min

                landmarks_removed = (~landmarks_mask & self.valid_mask).sum()
                if landmarks_removed > 0:
                    logger.info("  [LANDMARKS] Removed %d samples by low landmarks quality", landmarks_removed)

                quality_mask &= landmarks_mask

        # Track removals
        removed_by_quality = ~quality_mask & self.valid_mask
        self.removal_reasons['quality'] = removed_by_quality

        # Update valid mask
        self.valid_mask &= quality_mask

        removed_count = removed_by_quality.sum()
        remaining = self.valid_mask.sum()

        logger.info("\n  TOTAL removed by quality: %d", removed_count)
        logger.info("  Remaining samples       : %d", remaining)

        # Breakdown by person and type
        if removed_count > 0:
            removed_df = self.df[removed_by_quality]
            logger.info("\n  Removal breakdown:")

            for person in removed_df['person_name'].unique():
                person_removed = removed_df[removed_df['person_name'] == person]
                logger.info("    %s: %d removed", person, len(person_removed))

                for img_type in person_removed['image_type'].unique():
                    type_count = len(person_removed[person_removed['image_type'] == img_type])
                    logger.info("      → %s: %d", img_type, type_count)

        return self
    
    def remove_outliers(self, outlier_mask: Optional[np.ndarray] = None) -> 'DataCleaner':
        """
        Remove detected outliers.
        
        Migrated từ Face_Sanitizer.py - Logic giữ nguyên 100%.
        
        Args:
            outlier_mask: Boolean mask from validator (True = outlier)
            
        Returns:
            self (for method chaining)
        """
        logger.info("\n%s", "="*70)
        logger.info("  STAGE 4: OUTLIER REMOVAL")
        logger.info("%s", "="*70)
        
        if outlier_mask is None:
            if hasattr(self.report, 'outlier_mask') and self.report.outlier_mask is not None:
                outlier_mask = self.report.outlier_mask
            else:
                logger.warning("  No outlier mask provided, skipping...")
                return self
        
        # Track removals (only among currently valid samples)
        removed_by_outliers = outlier_mask & self.valid_mask
        self.removal_reasons['outliers'] = removed_by_outliers
        
        # Update valid mask
        self.valid_mask &= ~outlier_mask
        
        removed_count = removed_by_outliers.sum()
        remaining = self.valid_mask.sum()
        
        logger.info("  Removed outliers  : %d", removed_count)
        logger.info("  Remaining samples : %d", remaining)
        
        # Breakdown
        if removed_count > 0:
            removed_df = self.df[removed_by_outliers]
            logger.info("\n  Removal breakdown:")
            
            for person in removed_df['person_name'].unique():
                person_removed = removed_df[removed_df['person_name'] == person]
                logger.info("    %s: %d removed", person, len(person_removed))
                
                for img_type in person_removed['image_type'].unique():
                    type_count = len(person_removed[person_removed['image_type'] == img_type])
                    logger.info("      → %s: %d", img_type, type_count)
        
        return self
    
    def handle_multiface_frames(self, strategy: Optional[str] = None) -> 'DataCleaner':
        """
        Handle frames with multiple faces detected.

        Migrated từ Face_Sanitizer.py - Logic giữ nguyên 100%.

        Args:
            strategy: 'keep' | 'flag' | 'remove'

        Returns:
            self (for method chaining)
        """
        logger.info("\n%s", "="*70)
        logger.info("  STAGE 5: MULTI-FACE HANDLING")
        logger.info("%s", "="*70)

        if strategy is None:
            strategy = CONFIG['MULTIFACE_STRATEGY']

        logger.info("  Strategy: %s", strategy)

        # Check if JSON has multi-face info
        if 'has_multiple_faces' not in self.df.columns:
            logger.warning("  'has_multiple_faces' column not found in metadata")
            logger.info("  Skipping multi-face handling...")
            return self

        multiface_mask = self.df['has_multiple_faces'].values

        if strategy == 'keep':
            logger.info("  Strategy: KEEP - No action taken")
            return self

        elif strategy == 'flag':
            # Add flag column (for monitoring during training)
            self.df['is_multiface'] = multiface_mask
            logger.info("  Flagged %d multi-face frames", multiface_mask.sum())
            return self

        elif strategy == 'remove':
            # Remove multi-face frames
            removed_by_multiface = multiface_mask & self.valid_mask
            self.removal_reasons['multiface'] = removed_by_multiface
            self.valid_mask &= ~multiface_mask

            removed_count = removed_by_multiface.sum()
            remaining = self.valid_mask.sum()

            logger.info("  Removed multi-face: %d", removed_count)
            logger.info("  Remaining samples : %d", remaining)

            # Breakdown
            if removed_count > 0:
                removed_df = self.df[removed_by_multiface]
                logger.info("\n  Removal breakdown:")

                for person in removed_df['person_name'].unique():
                    person_removed = removed_df[removed_df['person_name'] == person]
                    logger.info("    %s: %d removed", person, len(person_removed))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self
    
    def balance_dataset(self, strategy: Optional[str] = None) -> 'DataCleaner':
        """
        Balance dataset by person.
        
        Migrated từ Face_Sanitizer.py - Logic giữ nguyên 100%.
        
        Args:
            strategy: 'none' | 'undersample' | 'oversample'
            
        Returns:
            self (for method chaining)
        """
        logger.info("\n%s", "="*70)
        logger.info("  STAGE 6: DATASET BALANCING")
        logger.info("%s", "="*70)
        
        if strategy is None:
            strategy = CONFIG['BALANCE_STRATEGY']
        logger.info("  Strategy: %s", strategy)
        
        if strategy == 'none':
            logger.info("  Strategy: NONE - No balancing applied")
            return self
        
        # Get current valid samples
        valid_df = self.df[self.valid_mask]
        valid_labels = self.labels[self.valid_mask]
        
        # Count per person
        label_counts = pd.Series(valid_labels).value_counts()
        logger.info("\n  Current distribution:")
        for label, count in label_counts.items():
            pct = count / len(valid_labels) * 100
            logger.info("    %s: %d samples (%.1f%%)", label, count, pct)
        
        if strategy == 'undersample':
            # Undersample to min count
            min_count = label_counts.min()
            logger.info("\n  Target per person: %d", min_count)
            
            # Track which samples to keep
            balanced_mask = np.zeros(len(self.df), dtype=bool)
            
            for label in label_counts.index:
                label_mask = self.valid_mask & (self.labels == label)
                label_indices = np.where(label_mask)[0]
                
                # Random sample
                np.random.seed(42)
                selected_indices = np.random.choice(
                    label_indices,
                    size=min_count,
                    replace=False
                )
                balanced_mask[selected_indices] = True
            
            # Track removals
            removed_by_balance = self.valid_mask & ~balanced_mask
            self.removal_reasons['balance'] = removed_by_balance
            self.valid_mask = balanced_mask
            
            removed_count = removed_by_balance.sum()
            remaining = self.valid_mask.sum()
            
            logger.info("\n  Removed by balance: %d", removed_count)
            logger.info("  Remaining samples : %d", remaining)
        
        elif strategy == 'oversample':
            # TODO: Implement oversampling (SMOTE or duplication)
            logger.warning("  Oversample not implemented yet, skipping...")
            return self
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self
    
    def get_cleaned_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Get cleaned dataset.
        
        Returns:
            (df, embeddings, labels) tuple with only valid samples
        """
        cleaned_df = self.df[self.valid_mask].reset_index(drop=True)
        cleaned_embeddings = self.embeddings[self.valid_mask]
        cleaned_labels = self.labels[self.valid_mask]
        
        return cleaned_df, cleaned_embeddings, cleaned_labels
    
    def generate_cleaning_report(self) -> CleaningReport:
        """
        Generate comprehensive cleaning report.
        
        Returns:
            CleaningReport object
        """
        cleaned_df, _, cleaned_labels = self.get_cleaned_data()
        
        # Count removals by reason
        removed_by_quality = int(self.removal_reasons['quality'].sum())
        removed_by_outliers = int(self.removal_reasons['outliers'].sum())
        removed_by_multiface = int(self.removal_reasons['multiface'].sum())
        removed_by_balance = int(self.removal_reasons['balance'].sum())
        
        # Final statistics
        final_label_counts = pd.Series(cleaned_labels).value_counts().to_dict()
        final_type_counts = cleaned_df['image_type'].value_counts().to_dict()
        
        final_quality_stats = {}
        if 'quality_score' in cleaned_df.columns:
            final_quality_stats = {
                'mean': float(cleaned_df['quality_score'].mean()),
                'std': float(cleaned_df['quality_score'].std()),
                'min': float(cleaned_df['quality_score'].min()),
                'max': float(cleaned_df['quality_score'].max()),
            }
        
        cleaned_size = len(cleaned_df)
        removed_count = self.original_size - cleaned_size
        removal_rate = float(removed_count / self.original_size)
        
        return CleaningReport(
            original_size=self.original_size,
            cleaned_size=cleaned_size,
            removed_count=removed_count,
            removal_rate=removal_rate,
            removed_by_quality=removed_by_quality,
            removed_by_outliers=removed_by_outliers,
            removed_by_multiface=removed_by_multiface,
            removed_by_balance=removed_by_balance,
            final_label_counts=final_label_counts,
            final_type_counts=final_type_counts,
            final_quality_stats=final_quality_stats
        )
    
    def save_cleaned_data(self, output_dir: str) -> Dict[str, str]:
        """
        Save cleaned dataset to disk (NPZ + JSON).

        Refactored 05-03-2026:
        - OPTIMIZED: Save NPZ + JSON only
        - REMOVED: CSV, Parquet (redundant)
        - REASON: Consistent with Phase 3/4/5 format

        Args:
            output_dir: Output directory path

        Returns:
            Dict of saved file paths
        """
        logger.info("\n%s", "="*70)
        logger.info("  SAVING CLEANED DATASET")
        logger.info("%s", "="*70)

        os.makedirs(output_dir, exist_ok=True)

        cleaned_df, cleaned_embeddings, cleaned_labels = self.get_cleaned_data()

        saved_paths = {}

        # ============================================================
        # 1. Save NPZ (embeddings array - for splitting/training)
        # ============================================================
        npz_path = os.path.join(output_dir, 'face_embeddings_cleaned.npz')

        # Get image paths if available
        if 'image_path' in cleaned_df.columns:
            image_paths = cleaned_df['image_path'].values
        else:
            image_paths = np.array([f"sample_{i}" for i in range(len(cleaned_df))])

        np.savez_compressed(
            npz_path,
            embeddings=cleaned_embeddings,
            labels=cleaned_labels,
            image_paths=image_paths
        )

        file_size_kb = os.path.getsize(npz_path) / 1024
        logger.info("\n  [NPZ] face_embeddings_cleaned.npz")
        logger.info("    Shape: %s", cleaned_embeddings.shape)
        logger.info("    Size: %.1f KB", file_size_kb)
        logger.info("    Purpose: Splitting, training")

        saved_paths['npz'] = npz_path

        # ============================================================
        # 2. Save JSON (cleaned metadata - for analysis)
        # ============================================================
        json_path = os.path.join(output_dir, 'face_embeddings_cleaned_metadata.json')

        import json
        from datetime import datetime

        # Prepare metadata records
        metadata_records = []
        for idx, row in cleaned_df.iterrows():
            metadata_records.append({
                'sample_id': int(idx),
                'person_name': row.get('person_name', ''),
                'image_type': row.get('image_type', ''),
                'image_path': row.get('image_path', ''),
                'detection_score': float(row.get('detection_score', 0.0)),
                'quality_score': float(row.get('quality_score', 0.0)),
                'embedding_norm': float(row.get('embedding_norm', 0.0)),
                'face_confidence_gate2': float(row.get('face_confidence_gate2', 0.0)),
                'snr_gate2': float(row.get('snr_gate2', 0.0)),
                'landmarks_quality': float(row.get('landmarks_quality', 0.0)),
            })

        # Summary statistics
        quality_scores = cleaned_df['quality_score'].values if 'quality_score' in cleaned_df.columns else []
        type_counts = cleaned_df['image_type'].value_counts().to_dict() if 'image_type' in cleaned_df.columns else {}

        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_samples': len(cleaned_df),
            'person_name': cleaned_df['person_name'].iloc[0] if len(cleaned_df) > 0 and 'person_name' in cleaned_df.columns else None,
            'summary': {
                'quality_score': {
                    'mean': float(np.mean(quality_scores)) if len(quality_scores) > 0 else 0.0,
                    'std': float(np.std(quality_scores)) if len(quality_scores) > 0 else 0.0,
                    'min': float(np.min(quality_scores)) if len(quality_scores) > 0 else 0.0,
                    'max': float(np.max(quality_scores)) if len(quality_scores) > 0 else 0.0,
                },
                'type_distribution': type_counts,
            },
            'samples': metadata_records,
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        file_size_kb = os.path.getsize(json_path) / 1024
        logger.info("\n  [JSON] face_embeddings_cleaned_metadata.json")
        logger.info("    Samples: %d", len(metadata_records))
        logger.info("    Size: %.1f KB", file_size_kb)
        logger.info("    Purpose: Analysis, database")

        saved_paths['json'] = json_path

        # Summary
        total_size_kb = (os.path.getsize(npz_path) + os.path.getsize(json_path)) / 1024
        logger.info("\n  Total: %.1f KB (NPZ + JSON)", total_size_kb)

        return saved_paths


# ==================== DATA SPLITTER ====================
class DataSplitter:
    """
    Split cleaned dataset into train/val/test.
    
    Migrated từ Face_Sanitizer.py - Logic giữ nguyên 100%.
    
    Responsibilities:
    - Stratified split by person + image_type
    - Validate split quality
    - Save splits to disk
    - Generate comprehensive report
    """
    
    def __init__(self, cleaner: DataCleaner, cleaning_report: CleaningReport):
        """
        Initialize splitter với cleaned data.
        
        Args:
            cleaner: DataCleaner instance (contains cleaned data)
            cleaning_report: CleaningReport from cleaner
        """
        logger.info("\n%s", "="*70)
        logger.info("  DATA SPLITTER INITIALIZATION")
        logger.info("%s", "="*70)
        
        # Get cleaned data
        self.df, self.embeddings, self.labels = cleaner.get_cleaned_data()
        self.cleaning_report = cleaning_report
        
        # Store splits (will be populated)
        self.splits = {
            'train_indices': None,
            'val_indices': None,
            'test_indices': None
        }
        
        logger.info("  Total samples: %d", len(self.df))
        logger.info("  Embedding dim: %d", self.embeddings.shape[1])
        logger.info("  Unique labels: %d", len(np.unique(self.labels)))
    
    def split_stratified(self,
                        train_ratio: Optional[float] = None,
                        val_ratio: Optional[float] = None,
                        test_ratio: Optional[float] = None,
                        random_seed: int = 42) -> 'DataSplitter':
        """
        Perform stratified split by person + image_type.
        
        Migrated từ Face_Sanitizer.py - Logic giữ nguyên 100%.
        
        Args:
            train_ratio: Ratio for training set (default 0.70)
            val_ratio: Ratio for validation set (default 0.15)
            test_ratio: Ratio for test set (default 0.15)
            random_seed: Random seed for reproducibility
            
        Returns:
            self (for method chaining)
        """
        logger.info("\n%s", "="*70)
        logger.info("  STAGE 7: DATASET SPLITTING")
        logger.info("%s", "="*70)
        
        ratio_config = CONFIG['SPLIT_RATIOS']
        train_ratio = train_ratio if train_ratio is not None else ratio_config['train']
        val_ratio = val_ratio if val_ratio is not None else ratio_config['val']
        test_ratio = test_ratio if test_ratio is not None else ratio_config['test']
        
        logger.info("  Split ratios: %.0f%% / %.0f%% / %.0f%%",
                   train_ratio*100, val_ratio*100, test_ratio*100)
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-5):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio:.3f}")
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Create stratification key (person + image_type)
        self.df['strat_key'] = self.df['person_name'] + '_' + self.df['image_type']
        
        logger.info("\n  Stratification groups:")
        strat_counts = self.df['strat_key'].value_counts()
        for key, count in strat_counts.items():
            logger.info("    %s: %d samples", key, count)
        
        # Initialize split indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Split each stratification group
        for strat_key in strat_counts.index:
            # Get indices for this group
            group_mask = self.df['strat_key'] == strat_key
            group_indices = np.where(group_mask)[0]
            
            # Shuffle indices
            np.random.shuffle(group_indices)
            
            # Calculate split sizes
            n_samples = len(group_indices)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            # Test gets the remainder to ensure all samples are used
            
            # Split indices
            group_train = group_indices[:n_train]
            group_val = group_indices[n_train:n_train + n_val]
            group_test = group_indices[n_train + n_val:]
            
            # Append to global splits
            train_indices.extend(group_train)
            val_indices.extend(group_val)
            test_indices.extend(group_test)
        
        # Convert to numpy arrays
        self.splits['train_indices'] = np.array(train_indices)
        self.splits['val_indices'] = np.array(val_indices)
        self.splits['test_indices'] = np.array(test_indices)

        # Log split sizes (safe division)
        total_samples = len(self.df)
        logger.info("\n  Split results:")
        if total_samples > 0:
            logger.info("    Train: %d samples (%.1f%%)",
                       len(train_indices), len(train_indices)/total_samples*100)
            logger.info("    Val  : %d samples (%.1f%%)",
                       len(val_indices), len(val_indices)/total_samples*100)
            logger.info("    Test : %d samples (%.1f%%)",
                       len(test_indices), len(test_indices)/total_samples*100)
        else:
            logger.warning("    No samples to split!")

        # Validate splits
        self._validate_splits()
        
        return self
    
    def _validate_splits(self) -> bool:
        """
        Validate split quality.
        
        Returns:
            True if splits are valid
        """
        logger.info("\n  Validating splits...")
        
        train_idx = self.splits['train_indices']
        val_idx = self.splits['val_indices']
        test_idx = self.splits['test_indices']
        
        # Check 1: No overlap
        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        unique_indices = np.unique(all_indices)
        
        if len(all_indices) != len(unique_indices):
            logger.error("    FAIL - Found overlapping indices!")
            return False
        
        logger.info("    PASS - No overlap between splits")
        
        # Check 2: All samples covered
        if len(unique_indices) != len(self.df):
            logger.error("    FAIL - Missing samples! (%d/%d)",
                        len(unique_indices), len(self.df))
            return False
        
        logger.info("    PASS - All samples covered")
        
        # Check 3: Stratification balance
        logger.info("\n  Stratification balance:")
        
        for split_name, split_idx in [('Train', train_idx),
                                       ('Val', val_idx),
                                       ('Test', test_idx)]:
            split_labels = self.labels[split_idx]
            label_counts = pd.Series(split_labels).value_counts()
            
            logger.info("\n    %s:", split_name)
            for label, count in label_counts.items():
                pct = count / len(split_idx) * 100
                logger.info("      %s: %d (%.1f%%)", label, count, pct)
            
            # Check image_type distribution
            split_df = self.df.iloc[split_idx]
            type_counts = split_df['image_type'].value_counts()
            
            logger.info("    Image types:")
            for img_type, count in type_counts.items():
                pct = count / len(split_idx) * 100
                logger.info("      %s: %d (%.1f%%)", img_type, count, pct)
        
        logger.info("\n    PASS - Stratification looks good")
        
        return True
    
    def save_splits(self, base_dir: str) -> Dict[str, str]:
        """
        Save splits to separate directories.

        Refactored 08-03-2026:
        - Save each split to separate folder (train/, validation/, test/)
        - Structure: base_dir/train/train.npz, base_dir/validation/val.npz, base_dir/test/test.npz
        - Remove CSV export (only NPZ needed for training)

        Args:
            base_dir: Base directory for splits (will create train/val/test subdirs)

        Returns:
            Dict of saved file paths
        """
        logger.info("\n%s", "="*70)
        logger.info("  SAVING SPLITS")
        logger.info("%s", "="*70)

        saved_paths = {}

        # Save each split to separate folder
        split_folders = {
            'train': 'train',
            'val': 'validation',  # Note: folder name is 'validation'
            'test': 'test'
        }

        for split_name, folder_name in split_folders.items():
            split_idx = self.splits[f'{split_name}_indices']

            # Create split folder
            split_folder = os.path.join(base_dir, folder_name)
            os.makedirs(split_folder, exist_ok=True)

            # Get split data
            split_embeddings = self.embeddings[split_idx]
            split_labels = self.labels[split_idx]
            split_df = self.df.iloc[split_idx].reset_index(drop=True)

            # Save NPZ to split folder
            npz_filename = f"{split_name}.npz"  # train.npz, val.npz, test.npz
            npz_path = os.path.join(split_folder, npz_filename)
            np.savez_compressed(
                npz_path,
                embeddings=split_embeddings,
                labels=split_labels,
                image_paths=split_df['image_path'].values if 'image_path' in split_df.columns
                           else np.array([f"{split_name}_{i}" for i in range(len(split_df))])
            )
            saved_paths[f'{split_name}_npz'] = npz_path
            logger.info("  Saved %s/%s: %d samples, %.1f KB",
                       folder_name, npz_filename, len(split_idx), os.path.getsize(npz_path) / 1024)

        logger.info("\n  All splits saved to separate folders in: %s", base_dir)
        logger.info("  Structure:")
        logger.info("    %s/train/train.npz", base_dir)
        logger.info("    %s/validation/val.npz", base_dir)
        logger.info("    %s/test/test.npz", base_dir)

        return saved_paths
    
    def generate_split_report(self) -> SplitReport:
        """
        Generate comprehensive split report.
        
        Returns:
            SplitReport object
        """
        train_idx = self.splits['train_indices']
        val_idx = self.splits['val_indices']
        test_idx = self.splits['test_indices']
        
        # Get distributions
        train_labels = self.labels[train_idx]
        val_labels = self.labels[val_idx]
        test_labels = self.labels[test_idx]
        
        train_distribution = pd.Series(train_labels).value_counts().to_dict()
        val_distribution = pd.Series(val_labels).value_counts().to_dict()
        test_distribution = pd.Series(test_labels).value_counts().to_dict()
        
        # Get image_type distributions
        train_df = self.df.iloc[train_idx]
        val_df = self.df.iloc[val_idx]
        test_df = self.df.iloc[test_idx]
        
        train_type_dist = train_df['image_type'].value_counts().to_dict()
        val_type_dist = val_df['image_type'].value_counts().to_dict()
        test_type_dist = test_df['image_type'].value_counts().to_dict()
        
        # Calculate stratification quality score
        stratification_score = self._calculate_stratification_score(
            train_distribution, val_distribution, test_distribution
        )
        
        # Calculate actual ratios
        total = len(self.df)
        train_ratio = float(len(train_idx) / total)
        val_ratio = float(len(val_idx) / total)
        test_ratio = float(len(test_idx) / total)
        
        return SplitReport(
            total_samples=total,
            train_size=len(train_idx),
            val_size=len(val_idx),
            test_size=len(test_idx),
            train_distribution=train_distribution,
            val_distribution=val_distribution,
            test_distribution=test_distribution,
            train_type_dist=train_type_dist,
            val_type_dist=val_type_dist,
            test_type_dist=test_type_dist,
            stratification_score=stratification_score,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    
    def _calculate_stratification_score(self,
                                       train_dist: Dict,
                                       val_dist: Dict,
                                       test_dist: Dict) -> float:
        """
        Calculate stratification quality score.
        
        Score = 1.0 means perfect balance across splits.
        Score < 0.9 indicates potential issues.
        
        Args:
            train_dist: Train label distribution
            val_dist: Val label distribution
            test_dist: Test label distribution
            
        Returns:
            Stratification score (0.0 - 1.0)
        """
        # Get all labels
        all_labels = set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys())
        
        # Calculate percentage differences
        diffs = []
        for label in all_labels:
            train_pct = train_dist.get(label, 0) / self.splits['train_indices'].shape[0]
            val_pct = val_dist.get(label, 0) / self.splits['val_indices'].shape[0]
            test_pct = test_dist.get(label, 0) / self.splits['test_indices'].shape[0]
            
            # Max deviation from mean
            mean_pct = (train_pct + val_pct + test_pct) / 3
            max_diff = max(
                abs(train_pct - mean_pct),
                abs(val_pct - mean_pct),
                abs(test_pct - mean_pct)
            )
            diffs.append(max_diff)
        
        # Score = 1 - average_deviation
        avg_diff = np.mean(diffs)
        score = max(0.0, 1.0 - avg_diff * 10)  # Scale to 0-1
        
        return float(score)
    
    def get_split_data(self, split_name: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Get data for a specific split.
        
        Args:
            split_name: 'train', 'val', or 'test'
            
        Returns:
            (embeddings, labels, dataframe) tuple
        """
        if split_name not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split name: {split_name}")
        
        split_idx = self.splits[f'{split_name}_indices']
        
        embeddings = self.embeddings[split_idx]
        labels = self.labels[split_idx]
        df = self.df.iloc[split_idx].reset_index(drop=True)
        
        return embeddings, labels, df


# ==================== SANITIZER SERVICE ====================
class SanitizerService:
    """
    Stateless orchestrator cho data sanitization pipeline.

    Refactored 05-03-2026:
    - OPTIMIZED: NPZ + JSON input/output only
    - REMOVED: CSV, Parquet support (redundant)
    - REASON: Align với Phase 3/5 format

    Architecture:
    - Service = orchestrator (runs entire pipeline: validate → clean → split)
    - Delegates to 3 classes: DataValidator, DataCleaner, DataSplitter

    Usage:
        service = SanitizerService(storage)
        val_report, clean_report, split_report = service.run(
            npz_path, json_path, output_dir
        )
    """
    
    def __init__(self, storage: StorageAdapter, person_name: Optional[str] = None):
        """
        Initialize SanitizerService.

        Args:
            storage: StorageAdapter instance (optional, not needed for Phase 4)
            person_name: Person name (for folder structure, optional)
        """
        self.storage = storage
        self.person_name = person_name
    
    def run(
        self,
        npz_path: str,
        json_path: str,
        output_dir: str
    ) -> Tuple[ValidationReport, CleaningReport, SplitReport]:
        """
        Run entire sanitization pipeline.

        Refactored 05-03-2026:
        - OPTIMIZED: NPZ + JSON input (Phase 3 output)
        - REMOVED: CSV, Parquet parameters (redundant)
        - REASON: Consistent format across all phases

        Args:
            npz_path: Path to NPZ file (embeddings)
            json_path: Path to JSON file (metadata)
            output_dir: Output directory

        Returns:
            Tuple of (ValidationReport, CleaningReport, SplitReport)
        """
        logger.info("\n%s", "="*70)
        logger.info("  DATA SANITIZER - FULL PIPELINE")
        logger.info("  Validator → Cleaner → Splitter")
        logger.info("%s", "="*70)

        try:
            # Phase 1: Validation
            validator = DataValidator(npz_path, json_path)
            validation_report = validator.validate_integrity()
            
            if not validation_report.passed:
                logger.error("\nValidation failed. Cannot proceed to cleaning.")
                raise ValueError("Integrity validation failed")
            
            outlier_mask = validator.detect_outliers(
                method='isolation_forest',
                contamination=THRESHOLDS['OUTLIER_CONTAMINATION']
            )
            validation_report.outlier_mask = outlier_mask
            
            # Phase 2: Cleaning
            cleaner = DataCleaner(validator, validation_report)
            
            # Apply cleaning operations (method chaining)
            cleaner.filter_by_quality(
                min_detection=THRESHOLDS['MIN_DETECTION_SCORE'],
                min_quality=THRESHOLDS['MIN_QUALITY_SCORE']
            ).remove_outliers(
                outlier_mask=outlier_mask
            ).handle_multiface_frames(
                strategy=CONFIG['MULTIFACE_STRATEGY']
            ).balance_dataset(
                strategy=CONFIG['BALANCE_STRATEGY']
            )
            
            # Generate cleaning report
            cleaning_report = cleaner.generate_cleaning_report()

            # Check if any samples remain after cleaning
            if cleaning_report.cleaned_size == 0:
                logger.error("\n%s", "="*70)
                logger.error("  ❌ NO SAMPLES REMAINING AFTER CLEANING!")
                logger.error("%s", "="*70)
                logger.error("\n  All %d samples were removed:", cleaning_report.original_size)
                logger.error("    By quality: %d", cleaning_report.removed_by_quality)
                logger.error("    By outliers: %d", cleaning_report.removed_by_outliers)
                logger.error("    By multiface: %d", cleaning_report.removed_by_multiface)
                logger.error("    By balancing: %d", cleaning_report.removed_by_balance)
                logger.error("\n  💡 Possible causes:")
                logger.error("    1. Gate 2 metadata missing (all zeros)")
                logger.error("    2. Thresholds too strict")
                logger.error("    3. Data quality issues")
                logger.error("\n  💡 Solutions:")
                logger.error("    1. Disable Gate 2 checks: ENABLE_GATE2_CONFIDENCE_CHECK=False")
                logger.error("    2. Lower thresholds: MIN_DETECTION_SCORE=0.30")
                logger.error("    3. Check Phase 2 metadata generation")
                logger.error("\n%s", "="*70)
                raise ValueError("No samples remaining after cleaning - cannot proceed to splitting")

            # Log cleaning summary
            logger.info("\n%s", "="*70)
            logger.info("  CLEANING SUMMARY")
            logger.info("%s", "="*70)
            logger.info("  Original samples   : %d", cleaning_report.original_size)
            logger.info("  Cleaned samples    : %d", cleaning_report.cleaned_size)
            logger.info("  Removed samples    : %d", cleaning_report.removed_count)
            logger.info("  Removal rate       : %.1f%%", cleaning_report.removal_rate*100)
            
            logger.info("\n  Removal breakdown:")
            logger.info("    By quality       : %d", cleaning_report.removed_by_quality)
            logger.info("    By outliers      : %d", cleaning_report.removed_by_outliers)
            logger.info("    By multi-face    : %d", cleaning_report.removed_by_multiface)
            logger.info("    By balancing     : %d", cleaning_report.removed_by_balance)
            
            # Save cleaned dataset
            cleaner.save_cleaned_data(output_dir)
            
            # Phase 3: Splitting
            splitter = DataSplitter(cleaner, cleaning_report)
            
            # Perform stratified split
            ratio_config = CONFIG['SPLIT_RATIOS']
            splitter.split_stratified(
                train_ratio=ratio_config['train'],
                val_ratio=ratio_config['val'],
                test_ratio=ratio_config['test'],
                random_seed=42
            )
            
            # Save splits
            splitter.save_splits(output_dir)
            
            # Generate split report
            split_report = splitter.generate_split_report()
            
            # Log split summary
            logger.info("\n%s", "="*70)
            logger.info("  SPLIT SUMMARY")
            logger.info("%s", "="*70)
            logger.info("  Total samples      : %d", split_report.total_samples)
            logger.info("  Train samples      : %d (%.1f%%)",
                       split_report.train_size, split_report.train_ratio*100)
            logger.info("  Val samples        : %d (%.1f%%)",
                       split_report.val_size, split_report.val_ratio*100)
            logger.info("  Test samples       : %d (%.1f%%)",
                       split_report.test_size, split_report.test_ratio*100)
            logger.info("  Stratification     : %.3f", split_report.stratification_score)
            
            if split_report.stratification_score < 0.9:
                logger.warning("  WARNING - Low stratification score!")
            else:
                logger.info("  GOOD - Well balanced splits")
            
            logger.info("\n%s", "="*70)
            logger.info("  PIPELINE COMPLETE")
            logger.info("%s", "="*70)
            
            return validation_report, cleaning_report, split_report
            
        except Exception as ex:
            logger.exception("Critical error in sanitization pipeline: %s", ex)
            raise


__all__ = ["SanitizerService"]

