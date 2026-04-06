"""Stateless Recognition Training Service - Production-ready.

Refactored 08-03-2026:
- Final phase: Training → MinIO + SQLite → Cleanup
- Upload vectors to MinIO
- Save metadata to SQLite
- Cleanup temp files after completion

Architecture:
- SimpleFaceRecognizer: Core recognition engine
- ThresholdTuner: Threshold optimization
- QualityGate: Quality assurance + user confirmation
- StorageAdapter: Upload NPZ to MinIO
- SQLiteManager: Save metadata to SQLite
- Cleanup: Delete data/temp/{person}/ after success

Workflow:
1. Load train.npz from Phase 4
2. Train SimpleFaceRecognizer
3. Optimize threshold (multi-threshold testing)
4. Quality gate (user confirmation)
5. Save database.npz
6. Upload NPZ to MinIO
7. Save metadata to SQLite
8. Cleanup: DELETE data/temp/{person}/
"""

import json
import logging
import os
import shutil
import uuid
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    RECOGNITION_CONFIG,
    RECOGNITION_QUALITY_THRESHOLDS,
    DATABASE_DIR,
    IOT_MINIO_KEYS,
    INFO, SUCCESS, ERROR, WARNING,
    console, create_table, print_section_header,
)
from core.adapters import StorageAdapter
from core.entities import EvaluationMetrics, RecognitionResult
from core.storage import get_sqlite_manager

logger = logging.getLogger(__name__)

# Load config
CONFIG = {
    'default_threshold': RECOGNITION_CONFIG['DEFAULT_THRESHOLD'],
    'test_thresholds': RECOGNITION_CONFIG['TEST_THRESHOLDS'],
}
QUALITY_GATE_CONFIG = RECOGNITION_CONFIG['QUALITY_GATE']
RAW_VIDEO_EXTENSIONS = RECOGNITION_CONFIG['RAW_VIDEO_EXTENSIONS']

QUALITY_THRESHOLDS = RECOGNITION_QUALITY_THRESHOLDS


# ==================== SIMPLE FACE RECOGNIZER ====================
class SimpleFaceRecognizer:
    """
    Simple face recognition system.
    
    Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.
    """
    
    def __init__(self, train_npz_path: str, threshold: float = 0.65):
        """
        Khởi tạo SimpleFaceRecognizer.

        Args:
            train_npz_path: Path to training NPZ file
            threshold: Cosine similarity threshold
        """
        print_section_header("FACE RECOGNIZER INITIALIZATION")

        self.threshold = threshold

        logger.info(f"\n{INFO} Loading training data...")
        logger.info(f"    File: {Path(train_npz_path).name}")

        train_data = np.load(train_npz_path)

        self.train_embeddings = train_data['embeddings']
        self.train_labels = train_data['labels']

        logger.info(f"{SUCCESS} Train embeddings: {self.train_embeddings.shape}")
        logger.info(f"{SUCCESS} Unique persons: {len(np.unique(self.train_labels))}")

        logger.info(f"\n{INFO} Building database...")
        self.database = self._build_database()

        # Database Contents Table
        db_rows = []
        for person, emb in self.database.items():
            count = np.sum(self.train_labels == person)
            db_rows.append([person, f"{count} samples", "512-dim center"])

        db_table = create_table(
            "Database Contents",
            ["Person", "Training Samples", "Embedding"],
            db_rows
        )
        console.print(db_table)

        logger.info(f"\n{INFO} Recognition threshold: {self.threshold:.2f}")
    
    def _build_database(self) -> Dict[str, np.ndarray]:
        """Build person database từ training embeddings."""
        database = {}
        
        for person in np.unique(self.train_labels):
            person_mask = self.train_labels == person
            person_embeddings = self.train_embeddings[person_mask]
            
            person_center = person_embeddings.mean(axis=0)
            person_center = person_center / np.linalg.norm(person_center)
            
            database[person] = person_center
        
        return database
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        similarity = np.dot(emb1, emb2)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def recognize(self, test_embedding: np.ndarray) -> RecognitionResult:
        """
        Recognize person từ test embedding.
        
        Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.
        
        Args:
            test_embedding: Test face embedding (512-dim)
            
        Returns:
            RecognitionResult object
        """
        similarities = {}
        for person, person_center in self.database.items():
            sim = self.cosine_similarity(test_embedding, person_center)
            similarities[person] = sim
        
        best_person = max(similarities, key=similarities.get)
        best_similarity = similarities[best_person]
        
        is_known = best_similarity >= self.threshold
        predicted_label = best_person if is_known else "Unknown"
        
        return RecognitionResult(
            predicted_label=predicted_label,
            true_label="",  # Will be filled by evaluate()
            confidence=best_similarity,
            similarities=similarities,
            is_correct=False,  # Will be filled by evaluate()
            is_known=is_known
        )
    
    def evaluate(self, test_npz_path: str) -> Tuple[List[RecognitionResult], EvaluationMetrics]:
        """
        Evaluate recognizer trên test set.

        Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.

        Args:
            test_npz_path: Path to test NPZ file

        Returns:
            Tuple of (results, metrics)
        """
        print_section_header("EVALUATING ON TEST SET")
        logger.info(f"{INFO} Threshold: {self.threshold:.2f} (cosine similarity)")

        test_data = np.load(test_npz_path)
        test_embeddings = test_data['embeddings']
        test_labels = test_data['labels']

        logger.info(f"{INFO} Test samples: {len(test_labels)}")
        logger.info(f"\n{INFO} Processing...")

        results = []
        for i, (test_emb, true_label) in enumerate(zip(test_embeddings, test_labels)):
            result = self.recognize(test_emb)
            result.true_label = true_label
            result.is_correct = (result.predicted_label == true_label)
            results.append(result)

            if (i + 1) % 20 == 0:
                logger.info(f"    Progress: {i+1}/{len(test_labels)} ({(i+1)/len(test_labels)*100:.1f}%)")

        logger.info(f"{SUCCESS} Completed: {len(test_labels)}/{len(test_labels)}")

        metrics = self._calculate_metrics(results)
        return results, metrics
    
    def _calculate_metrics(self, results: List[RecognitionResult]) -> EvaluationMetrics:
        """Calculate evaluation metrics từ results."""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = float(correct / total) if total > 0 else 0.0
        
        known_results = [r for r in results if r.is_known]
        known_samples = len(known_results)
        known_correct = sum(1 for r in known_results if r.is_correct)
        known_accuracy = float(known_correct / known_samples) if known_samples > 0 else 0.0
        
        per_person_accuracy = {}
        for person in np.unique([r.true_label for r in results]):
            person_results = [r for r in results if r.true_label == person]
            person_correct = sum(1 for r in person_results if r.is_correct)
            per_person_accuracy[person] = float(person_correct / len(person_results))
        
        confusion_matrix = self._build_confusion_matrix(results)
        
        return EvaluationMetrics(
            total_samples=total,
            accuracy=accuracy,
            known_samples=known_samples,
            known_correct=known_correct,
            known_accuracy=known_accuracy,
            per_person_accuracy=per_person_accuracy,
            confusion_matrix=confusion_matrix,
            threshold=self.threshold
        )
    
    def _build_confusion_matrix(self, results: List[RecognitionResult]) -> Dict[str, Dict[str, int]]:
        """Build confusion matrix từ results."""
        true_labels = set(r.true_label for r in results)
        pred_labels = set(r.predicted_label for r in results)
        all_labels = true_labels | pred_labels
        
        matrix = {true: {pred: 0 for pred in all_labels} for true in true_labels}
        
        for result in results:
            matrix[result.true_label][result.predicted_label] += 1
        
        return matrix
    
    def print_evaluation_report(self, metrics: EvaluationMetrics):
        """
        Print detailed evaluation report.

        Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.
        """
        print_section_header("EVALUATION REPORT")

        # Overall Performance Table
        overall_table = create_table(
            "Overall Performance",
            ["Metric", "Value"],
            [
                ["Total samples", f"{metrics.total_samples}"],
                ["Correct", f"{int(metrics.accuracy * metrics.total_samples)}"],
                ["Accuracy", f"{metrics.accuracy * 100:.2f}%"],
            ]
        )
        console.print(overall_table)

        # Known Faces Only Table
        known_table = create_table(
            "Known Faces Only",
            ["Metric", "Value"],
            [
                ["Samples", f"{metrics.known_samples}"],
                ["Correct", f"{metrics.known_correct}"],
                ["Accuracy", f"{metrics.known_accuracy * 100:.2f}%"],
            ]
        )
        console.print(known_table)

        # Per-Person Accuracy
        logger.info(f"\n{INFO} Per-Person Accuracy:")
        for person, acc in sorted(metrics.per_person_accuracy.items()):
            status = SUCCESS if acc >= 0.75 else WARNING
            logger.info(f"  {status} {person.ljust(12)}: {acc * 100:.2f}%")

        logger.info("\n  Confusion Matrix:")
        all_pred_labels = set()
        for true_label in metrics.confusion_matrix:
            all_pred_labels.update(metrics.confusion_matrix[true_label].keys())

        header = "    {}".format("True \\ Pred".ljust(15))
        for pred_label in sorted(all_pred_labels):
            header += " {}".format(str(pred_label).rjust(9))
        logger.info(header)
        logger.info("    " + "-" * (len(header) - 4))

        for true_label in sorted(metrics.confusion_matrix.keys()):
            row = "    {}".format(str(true_label).ljust(15))
            for pred_label in sorted(all_pred_labels):
                count = metrics.confusion_matrix[true_label].get(pred_label, 0)
                row += " {}".format(str(count).rjust(9))
            logger.info(row)

        logger.info("%s", "="*70)
    
    def save_database(self, output_path: str) -> bool:
        """
        Save database to NPZ file.

        Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.

        Args:
            output_path: Output path for database

        Returns:
            True if successful
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if output_path.endswith('.npz'):
                db_dict = {}
                for person, embedding in self.database.items():
                    db_dict[person] = embedding.astype(np.float32)

                np.savez_compressed(output_path, **db_dict)

                file_size_kb = os.path.getsize(output_path) / 1024
                logger.info(f"\n{SUCCESS} Database saved:")
                logger.info(f"    File: {Path(output_path).name}")
                logger.info(f"    Size: {file_size_kb:.1f} KB")
                logger.info(f"    Persons: {len(db_dict)}")

            return True

        except Exception as ex:
            logger.exception(f"{ERROR} Failed to save database: {ex}")
            return False


# ==================== THRESHOLD TUNER ====================
class ThresholdTuner:
    """
    Threshold optimization tool.
    
    Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.
    """
    
    def __init__(self, train_npz_path: str):
        """
        Khởi tạo ThresholdTuner.
        
        Args:
            train_npz_path: Path to training NPZ file
        """
        self.train_npz_path = train_npz_path
    
    def tune(self, test_npz_path: str, thresholds: Optional[List[float]] = None) -> Dict[float, EvaluationMetrics]:
        """
        Test multiple thresholds và tìm best.

        Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.

        Args:
            test_npz_path: Path to test NPZ file
            thresholds: List of thresholds to test

        Returns:
            Dict mapping threshold → metrics
        """
        if thresholds is None:
            thresholds = CONFIG['test_thresholds']

        print_section_header("THRESHOLD TUNING")
        logger.info(f"{INFO} Testing {len(thresholds)} thresholds: {thresholds}")
        logger.info(f"{INFO} Strategy: Find threshold with highest overall accuracy")

        results = {}

        for idx, threshold in enumerate(thresholds, 1):
            logger.info("\n" + "-"*70)
            logger.info(f"{INFO} [{idx}/{len(thresholds)}] Testing threshold: {threshold:.2f}")
            logger.info("-"*70)

            recognizer = SimpleFaceRecognizer(self.train_npz_path, threshold=threshold)
            _, metrics = recognizer.evaluate(test_npz_path)
            results[threshold] = metrics

            logger.info(f"\n{INFO} Results:")
            logger.info(f"    Overall Accuracy: {metrics.accuracy * 100:.2f}%")
            logger.info(f"    Known Accuracy  : {metrics.known_accuracy * 100:.2f}%")

        self._print_comparison(results)
        return results
    
    def _print_comparison(self, results: Dict[float, EvaluationMetrics]):
        """Print comparison table for all thresholds."""
        print_section_header("THRESHOLD COMPARISON")

        # Get all person names
        all_persons = set()
        for metrics in results.values():
            all_persons.update(metrics.per_person_accuracy.keys())

        # Find best threshold
        best_threshold = None
        best_accuracy = 0.0
        for threshold in sorted(results.keys()):
            if results[threshold].accuracy > best_accuracy:
                best_accuracy = results[threshold].accuracy
                best_threshold = threshold

        # Build table rows
        table_rows = []
        for threshold in sorted(results.keys()):
            metrics = results[threshold]

            row = [
                f"{threshold:.2f}",
                f"{metrics.accuracy * 100:.2f}%",
                f"{metrics.known_accuracy * 100:.2f}%",
            ]

            # Add per-person accuracies
            for person in sorted(all_persons):
                acc = metrics.per_person_accuracy.get(person, 0.0)
                row.append(f"{acc * 100:.2f}%")

            # Mark best
            if threshold == best_threshold:
                row.append("BEST")
            else:
                row.append("")

            table_rows.append(row)

        # Create columns
        columns = ["Threshold", "Overall", "Known"] + sorted(all_persons) + [""]

        # Create table
        comparison_table = create_table(
            "Threshold Comparison",
            columns,
            table_rows
        )
        console.print(comparison_table)

        logger.info(f"\n{SUCCESS} Best Threshold: {best_threshold:.2f}")
        logger.info(f"    Overall Accuracy: {results[best_threshold].accuracy * 100:.2f}%")
        logger.info(f"    Known Accuracy: {results[best_threshold].known_accuracy * 100:.2f}%")


# ==================== QUALITY GATE ====================
class QualityGate:
    """
    Quality assessment và user confirmation.
    
    Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.
    """
    
    def __init__(self):
        """Khởi tạo QualityGate."""
        logger.info("\nQualityGate initialized")
    
    def generate_report(self, best_metrics: EvaluationMetrics, best_threshold: float,
                       all_metrics: Dict[float, EvaluationMetrics]) -> Dict:
        """
        Generate comprehensive quality report.
        
        Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.
        
        Args:
            best_metrics: Metrics của best threshold
            best_threshold: Best threshold value
            all_metrics: All threshold metrics
            
        Returns:
            Quality report dict
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_quality': self._assess_quality(best_metrics),
            'metrics_summary': {
                'best_threshold': float(best_threshold),
                'overall_accuracy': float(best_metrics.accuracy),
                'known_accuracy': float(best_metrics.known_accuracy),
                'test_samples': best_metrics.total_samples,
                'known_samples': best_metrics.known_samples,
                'per_person_accuracy': {k: float(v) for k, v in best_metrics.per_person_accuracy.items()}
            },
            'threshold_analysis': {
                float(t): {'accuracy': float(m.accuracy), 'known_accuracy': float(m.known_accuracy)}
                for t, m in all_metrics.items()
            },
            'confusion_matrix': best_metrics.confusion_matrix,
            'quality_checks': self._run_quality_checks(best_metrics),
            'warnings': self._generate_warnings(best_metrics),
            'recommendation': self._get_recommendation(best_metrics)
        }
        return report
    
    def _assess_quality(self, metrics: EvaluationMetrics) -> str:
        """Assess overall model quality."""
        excellent = QUALITY_GATE_CONFIG['EXCELLENT']
        good = QUALITY_GATE_CONFIG['GOOD']
        
        if metrics.accuracy >= excellent['overall'] and metrics.known_accuracy >= excellent['known']:
            return "EXCELLENT"
        elif metrics.accuracy >= good['overall'] and metrics.known_accuracy >= good['known']:
            return "GOOD"
        elif (metrics.accuracy >= QUALITY_THRESHOLDS['MIN_OVERALL_ACCURACY'] and
              metrics.known_accuracy >= QUALITY_THRESHOLDS['MIN_KNOWN_ACCURACY']):
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def _run_quality_checks(self, metrics: EvaluationMetrics) -> Dict[str, bool]:
        """Run all quality checks."""
        return {
            'overall_accuracy': metrics.accuracy >= QUALITY_THRESHOLDS['MIN_OVERALL_ACCURACY'],
            'known_accuracy': metrics.known_accuracy >= QUALITY_THRESHOLDS['MIN_KNOWN_ACCURACY'],
            'test_samples': metrics.total_samples >= QUALITY_THRESHOLDS['MIN_TEST_SAMPLES'],
            'per_person_accuracy': all(
                acc >= QUALITY_THRESHOLDS['MIN_PER_PERSON_ACCURACY']
                for acc in metrics.per_person_accuracy.values()
            )
        }
    
    def _generate_warnings(self, metrics: EvaluationMetrics) -> List[str]:
        """Generate warning messages."""
        warnings = []
        
        if metrics.accuracy < QUALITY_THRESHOLDS['WARN_OVERALL_ACCURACY']:
            warnings.append(
                f"WARNING: Overall accuracy thấp: {metrics.accuracy:.2%} "
                f"(khuyến nghị: >= {QUALITY_THRESHOLDS['WARN_OVERALL_ACCURACY']:.0%})"
            )
        
        if metrics.known_accuracy < QUALITY_THRESHOLDS['WARN_KNOWN_ACCURACY']:
            warnings.append(
                f"WARNING: Known faces accuracy thấp: {metrics.known_accuracy:.2%} "
                f"(khuyến nghị: >= {QUALITY_THRESHOLDS['WARN_KNOWN_ACCURACY']:.0%})"
            )
        
        for person, acc in metrics.per_person_accuracy.items():
            if acc < QUALITY_THRESHOLDS['MIN_PER_PERSON_ACCURACY']:
                warnings.append(
                    f"WARNING: {person} accuracy thấp: {acc:.2%} "
                    f"(khuyến nghị: >= {QUALITY_THRESHOLDS['MIN_PER_PERSON_ACCURACY']:.0%})"
                )
        
        sample_multiplier = QUALITY_GATE_CONFIG.get('SAMPLE_MULTIPLIER', 2)
        if metrics.total_samples < QUALITY_THRESHOLDS['MIN_TEST_SAMPLES'] * sample_multiplier:
            warnings.append(
                f"WARNING: Test set nhỏ: {metrics.total_samples} samples "
                f"(nên >= {QUALITY_THRESHOLDS['MIN_TEST_SAMPLES'] * sample_multiplier})"
            )
        
        return warnings
    
    def _get_recommendation(self, metrics: EvaluationMetrics) -> str:
        """Get recommendation based on quality."""
        quality = self._assess_quality(metrics)
        
        if quality == "EXCELLENT":
            return "RECOMMENDATION: LƯU DATABASE - Mô hình hoạt động xuất sắc"
        elif quality == "GOOD":
            return "RECOMMENDATION: LƯU DATABASE - Mô hình hoạt động tốt"
        elif quality == "ACCEPTABLE":
            return "RECOMMENDATION: CÓ THỂ LƯU - Nhưng nên kiểm tra lại"
        else:
            return "RECOMMENDATION: KHÔNG LƯU - Mô hình cần cải thiện"
    
    def can_save(self, metrics: EvaluationMetrics) -> bool:
        """Check if model can be saved automatically."""
        checks = self._run_quality_checks(metrics)
        return all(checks.values())
    
    def display_report(self, report: Dict):
        """Display quality gate report."""
        print_section_header("QUALITY GATE REPORT")

        quality = report['model_quality']
        if quality == "EXCELLENT":
            quality_status = f"{SUCCESS} EXCELLENT"
        elif quality == "GOOD":
            quality_status = f"{SUCCESS} GOOD"
        elif quality == "ACCEPTABLE":
            quality_status = f"{WARNING} ACCEPTABLE"
        else:
            quality_status = f"{ERROR} POOR"

        logger.info(f"\n{INFO} Model Quality: {quality_status}")
        logger.info(f"{INFO} {report['recommendation']}")

        metrics = report['metrics_summary']

        # Metrics Summary Table
        metrics_table = create_table(
            "Metrics Summary",
            ["Metric", "Value", "Minimum"],
            [
                ["Best Threshold", f"{metrics['best_threshold']:.2f}", "N/A"],
                ["Overall Accuracy", f"{metrics['overall_accuracy'] * 100:.2f}%", 
                 f"{QUALITY_THRESHOLDS['MIN_OVERALL_ACCURACY'] * 100:.0f}%"],
                ["Known Faces Accuracy", f"{metrics['known_accuracy'] * 100:.2f}%", 
                 f"{QUALITY_THRESHOLDS['MIN_KNOWN_ACCURACY'] * 100:.0f}%"],
                ["Test Samples", f"{metrics['test_samples']}", 
                 f"{QUALITY_THRESHOLDS['MIN_TEST_SAMPLES']}"],
                ["Known Samples", f"{metrics['known_samples']}", "N/A"],
            ]
        )
        console.print(metrics_table)

        logger.info(f"\n{INFO} Per-Person Accuracy:")
        for person, acc in sorted(metrics['per_person_accuracy'].items()):
            min_acc = QUALITY_THRESHOLDS['MIN_PER_PERSON_ACCURACY']
            if acc >= min_acc:
                status = f"{SUCCESS} PASS"
            else:
                status = f"{ERROR} FAIL"
            logger.info(f"  {status} {person.ljust(12)}: {acc * 100:.2f}% (min: {min_acc * 100:.0f}%)")

        logger.info(f"\n{INFO} Quality Checks:")
        checks = report['quality_checks']
        for check_name, passed in checks.items():
            status = SUCCESS if passed else ERROR
            logger.info(f"  {status} {check_name.replace('_', ' ').title()}")

        if report['warnings']:
            logger.info(f"\n{WARNING} Warnings:")
            for warning in report['warnings']:
                logger.info(f"    {warning}")

        logger.info(f"\n{INFO} Threshold Analysis:")
        logger.info(f"    {'Threshold'.center(10)} | {'Overall'.center(12)} | {'Known'.center(12)}")
        logger.info("    " + "-"*40)

        for threshold in sorted(report['threshold_analysis'].keys()):
            metrics_data = report['threshold_analysis'][threshold]
            marker = " BEST" if threshold == metrics['best_threshold'] else ""

            # Format values separately to avoid nested f-strings
            threshold_str = f"{threshold:.2f}".center(10)
            accuracy_str = f"{metrics_data['accuracy'] * 100:.2f}%".center(12)
            known_str = f"{metrics_data['known_accuracy'] * 100:.2f}%".center(12)

            logger.info(f"    {threshold_str} | {accuracy_str} | {known_str}{marker}")
    
    def ask_user_confirmation(self, report: Dict, interactive: bool = True) -> bool:
        """
        Ask user for confirmation to save database.

        Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.

        Args:
            report: Quality report dict
            interactive: If False, auto-save if quality is EXCELLENT/GOOD

        Returns:
            True if user confirms (or auto-approved)
        """
        quality = report['model_quality']
        recommendation = report['recommendation']

        print("\n" + "="*70)
        print("   SAVE CONFIRMATION")
        print("="*70)

        # Quality badge
        if quality == "EXCELLENT":
            print("\n  Model Quality:  {}".format(quality))
        elif quality == "GOOD":
            print("\n  Model Quality:  {}".format(quality))
        elif quality == "ACCEPTABLE":
            print("\n  Model Quality:   {}".format(quality))
        else:
            print("\n  Model Quality:  {}".format(quality))

        print("  {}".format(recommendation))

        metrics = report['metrics_summary']
        print("\n  Key Metrics:")
        print("    Overall Accuracy: {:.2f}%".format(metrics['overall_accuracy'] * 100))
        print("    Known Accuracy  : {:.2f}%".format(metrics['known_accuracy'] * 100))
        print("    Best Threshold  : {:.2f}".format(metrics['best_threshold']))

        if report['warnings']:
            print("\n    Warnings:")
            for warning in report['warnings']:
                print("    • {}".format(warning))

        print("\n" + "-"*70)

        # ✅ FIX: Auto-save if non-interactive and quality is good
        if not interactive:
            if quality in ["EXCELLENT", "GOOD"]:
                print("\n   Non-interactive mode: AUTO-SAVE (quality: {})".format(quality))
                return True
            else:
                print("\n    Non-interactive mode: SKIPPING SAVE (quality: {})".format(quality))
                print("     Recommendation: Review metrics manually")
                return False

        # Interactive mode: Ask user
        try:
            while True:
                response = input("\nDo you want to save the database? (Y/N): ").strip().lower()

                if response in ['yes', 'y']:
                    return True
                elif response in ['no', 'n']:
                    return False
                else:
                    print("Please enter 'yes' or 'no'")
        except (EOFError, KeyboardInterrupt):
            # Handle input errors gracefully
            print("\n\n    Input interrupted. Auto-deciding based on quality...")
            if quality in ["EXCELLENT", "GOOD"]:
                print("     Quality is {}: AUTO-SAVE".format(quality))
                return True
            else:
                print("     Quality is {}: SKIPPING SAVE".format(quality))
                return False


class RecognitionService:
    """
    Stateless orchestrator cho complete training workflow.
    
    Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.
    
    Architecture:
    - Service = orchestrator (runs entire pipeline)
    - SimpleFaceRecognizer = core recognition
    - ThresholdTuner = optimization
    - QualityGate = quality assurance
    - StorageAdapter = MinIO upload
    - SQLiteManager = metadata storage

    Usage:
        service = RecognitionService(storage, person_name="Nghi")
        recognizer, results, metrics, report = service.run(
            train_path, test_path, interactive=True
        )
    """
    
    def __init__(self, storage: StorageAdapter, person_name: Optional[str] = None):
        """
        Initialize RecognitionService.

        Args:
            storage: StorageAdapter instance (for MinIO upload)
            person_name: Person name (for cleanup after training)
        """
        self.storage = storage
        self.person_name = person_name
    
    def run(
        self,
        train_npz_path: str,
        test_npz_path: str,
        base_data_dir: Optional[Path] = None,
        database_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        interactive: bool = True
    ) -> Tuple[Optional[SimpleFaceRecognizer], List[RecognitionResult],
               EvaluationMetrics, Dict]:
        """
        Run complete training workflow.
        
        Migrated từ Face_Recognition.py - Logic giữ nguyên 100%.
        
        Args:
            train_npz_path: Path to train NPZ file
            test_npz_path: Path to test NPZ file
            base_data_dir: Base data directory for migration
            database_dir: Database directory for sync
            output_dir: Output directory for results
            interactive: Enable user confirmation
            
        Returns:
            Tuple of (recognizer, results, metrics, tuning_results)
        """
        print_section_header("FACE RECOGNITION - COMPLETE WORKFLOW")
        
        # Phase 1: Training & Evaluation
        recognizer = SimpleFaceRecognizer(train_npz_path, threshold=CONFIG['default_threshold'])
        results, metrics = recognizer.evaluate(test_npz_path)
        recognizer.print_evaluation_report(metrics)
        
        # Phase 2: Threshold Tuning
        tuner = ThresholdTuner(train_npz_path)
        tuning_results = tuner.tune(test_npz_path, CONFIG['test_thresholds'])
        
        best_threshold = max(tuning_results.keys(), key=lambda t: tuning_results[t].accuracy)
        best_metrics = tuning_results[best_threshold]
        
        logger.info("\n" + "="*70)
        logger.info(f"{SUCCESS} BEST THRESHOLD FOUND")
        logger.info("="*70)
        logger.info(f"{INFO} Threshold: {best_threshold:.2f}")
        logger.info(f"{INFO} Accuracy : {best_metrics.accuracy * 100:.2f}%")
        logger.info(f"{INFO} Known Acc: {best_metrics.known_accuracy * 100:.2f}%")
        logger.info("="*70)
        
        # Phase 3: Quality Gate
        quality_gate = QualityGate()
        report = quality_gate.generate_report(best_metrics, best_threshold, tuning_results)
        quality_gate.display_report(report)

        if interactive:
            should_save = quality_gate.ask_user_confirmation(report, interactive=True)
        else:
            should_save = quality_gate.ask_user_confirmation(report, interactive=False)
            logger.info("\n  Auto-save decision: %s", should_save)

        if not should_save:
            logger.warning("\n" + "="*70)
            logger.warning(f"{WARNING} USER REJECTED - Training cancelled")
            logger.warning("="*70)
            return None, results, best_metrics, tuning_results
        
        # Phase 4: Save Database
        print_section_header("SAVING TRAINED DATABASE")

        best_recognizer = SimpleFaceRecognizer(train_npz_path, threshold=best_threshold)

        # Save to output_dir if specified, otherwise use DATABASE_DIR (not REPORTS!)
        if output_dir is None:
            from config.settings import DATABASE_DIR
            output_dir = DATABASE_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        db_output_npz = output_dir / 'face_recognition_db.npz'

        if not best_recognizer.save_database(str(db_output_npz)):
            logger.error(f"\n{ERROR} Failed to save database file")
            return None, results, best_metrics, tuning_results
        
        # Save metadata
        metadata = {
            'best_threshold': float(best_threshold),
            'overall_accuracy': float(best_metrics.accuracy),
            'known_accuracy': float(best_metrics.known_accuracy),
            'per_person_accuracy': {k: float(v) for k, v in best_metrics.per_person_accuracy.items()},
            'total_samples': best_metrics.total_samples,
            'confusion_matrix': {k: {kk: int(vv) for kk, vv in v.items()}
                                for k, v in best_metrics.confusion_matrix.items()},
            'timestamp': datetime.now().isoformat(),
            'quality_report': report,
        }
        
        metadata_file = output_dir / 'face_recognition_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"\n{SUCCESS} Metadata saved:")
        logger.info(f"    File: {metadata_file.name}")
        logger.info(f"    Size: {metadata_file.stat().st_size / 1024:.1f} KB")

        # Phase 5: Upload to MinIO
        print_section_header("UPLOADING TO MINIO")

        try:
            # Upload aggregated recognition DB to the single canonical key
            storage_key = IOT_MINIO_KEYS['DATABASE_NPZ']

            # Merge with existing MinIO recognition DB to avoid overwriting other persons
            merged_db = {}

            existing_bytes = self.storage.get(storage_key, use_cache=False)
            if existing_bytes:
                existing_npz = np.load(io.BytesIO(existing_bytes), allow_pickle=False)
                for person_name in existing_npz.files:
                    merged_db[person_name] = existing_npz[person_name].astype(np.float32)

            # Update/insert persons from current training output
            for person_name, emb in best_recognizer.database.items():
                merged_db[person_name] = emb.astype(np.float32)

            # Save merged DB back to local output and upload
            np.savez_compressed(db_output_npz, **merged_db)

            if self.storage.put_file(db_output_npz, storage_key):
                logger.info(f"\n{SUCCESS} Uploaded to MinIO:")
                logger.info(f"    Key: {storage_key}")
                logger.info(f"    Persons in merged DB: {len(merged_db)}")
            else:
                logger.error(f"\n{ERROR} Failed to upload to MinIO")
                return None, results, best_metrics, tuning_results

        except Exception as ex:
            logger.error(f"\n{ERROR} MinIO upload failed: {ex}")
            return None, results, best_metrics, tuning_results

        # Phase 6: Save to SQLite
        print_section_header("SAVING TO SQLITE")

        try:
            sqlite_manager = get_sqlite_manager()

            # Add person if not exists
            person_id = None
            person = sqlite_manager.get_person_by_name(self.person_name)

            if person:
                person_id = person['person_id']
                logger.info(f"\n{INFO} Person exists: {self.person_name}")
            else:
                person_id = sqlite_manager.add_person(
                    name=self.person_name,
                    full_name=self.person_name,
                    vector_storage_key=storage_key,
                    vector_count=len(best_recognizer.database),
                    metadata={
                        'threshold': float(best_threshold),
                        'accuracy': float(best_metrics.accuracy),
                        'known_accuracy': float(best_metrics.known_accuracy),
                        'quality': report['model_quality'],
                        'trained_at': metadata['timestamp']
                    }
                )
                logger.info(f"\n{SUCCESS} Person added: {self.person_name} (ID: {person_id[:8]})")

            # Update vector info
            sqlite_manager.update_person(
                person_id,
                vector_storage_key=storage_key,
                vector_count=len(best_recognizer.database),
                metadata={
                    'threshold': float(best_threshold),
                    'accuracy': float(best_metrics.accuracy),
                    'known_accuracy': float(best_metrics.known_accuracy),
                    'quality': report['model_quality'],
                    'trained_at': metadata['timestamp']
                }
            )

            logger.info(f"{SUCCESS} Metadata saved to SQLite")
            logger.info(f"    Person ID: {person_id[:8]}")
            logger.info(f"    Storage Key: {storage_key}")

        except Exception as ex:
            logger.error(f"\n{ERROR} SQLite save failed: {ex}")
            logger.exception(f"{ERROR} Stack trace:")
            return None, results, best_metrics, tuning_results

        # Phase 7: Cleanup Temp Files
        if self.person_name:
            print_section_header("CLEANING UP TEMP FILES")

            try:
                from config.settings import get_person_temp_dir
                import shutil

                person_temp = get_person_temp_dir(self.person_name)

                if person_temp.exists():
                    shutil.rmtree(person_temp)
                    logger.info(f"\n{SUCCESS} Deleted temp folder:")
                    logger.info(f"    Path: {person_temp}")
                    logger.info(f"{INFO} This includes all videos, frames, vectors")
                else:
                    logger.warning(f"{WARNING} Temp folder not found: {person_temp}")

            except Exception as ex:
                logger.error(f"\n{ERROR} Cleanup failed: {ex}")
                logger.warning(f"{WARNING} Manual cleanup needed: data/temp/{self.person_name}/")

        # Final Summary
        print_section_header("TRAINING & DEPLOYMENT COMPLETE")

        logger.info(f"\n{INFO} Storage:")
        logger.info(f"    MinIO       : {storage_key}")
        logger.info(f"    SQLite      : Person {self.person_name} (ID: {person_id[:8]})")
        logger.info(f"    Local copy  : {db_output_npz}")
        logger.info(f"\n{INFO} Model Performance:")
        logger.info(f"    Quality     : {report['model_quality']}")
        logger.info(f"    Threshold   : {best_threshold:.2f}")
        logger.info(f"    Accuracy    : {best_metrics.accuracy * 100:.2f}%")
        logger.info(f"\n{SUCCESS} PIPELINE COMPLETE")
        logger.info(f"    {SUCCESS} Phase 1-4: Data -> Frames -> Embeddings -> Splits")
        logger.info(f"    {SUCCESS} Phase 5: Training -> MinIO + SQLite")
        logger.info(f"    {SUCCESS} Cleanup: Temp files deleted")
        logger.info(f"\n{INFO} Next Steps:")
        logger.info("    - Test realtime recognition")
        logger.info("    - Deploy to Raspberry Pi")
        
        return best_recognizer, results, best_metrics, tuning_results
    
    def create_recognizer(
        self,
        train_npz_path: str,
        threshold: Optional[float] = None
    ) -> SimpleFaceRecognizer:
        """
        Create face recognizer từ training data (simple mode).
        
        Args:
            train_npz_path: Path to training NPZ file
            threshold: Custom threshold (default from config)
            
        Returns:
            SimpleFaceRecognizer instance
        """
        threshold = threshold or CONFIG['default_threshold']
        return SimpleFaceRecognizer(train_npz_path, threshold)


__all__ = ["RecognitionService", "SimpleFaceRecognizer", "ThresholdTuner", "QualityGate"]

