"""Fine-tune service for incremental training (CPU-friendly, research-focused).

This service performs lightweight head training on ArcFace embeddings, logs full
training progress, exports training curves, and prepares replay samples.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from config.settings import (
    FINE_TUNE_CONFIG,
    EMBEDDING_PATHS,
    DATABASE_DIR,
    get_person_sanitized_dir,
    IOT_MINIO_KEYS,
    INFO,
    SUCCESS,
    WARNING,
    ERROR,
    METADATA,
)
from core.adapters.ModelAdapter import ModelAdapter
from core.adapters.FineTuneAdapter import FineTuneAdapter
from core.adapters.StorageAdapter import StorageAdapter
from core.entities import FineTuneEpochMetrics, FineTuneReport
from core.storage import get_sqlite_manager

logger = logging.getLogger(__name__)


def _to_portable_path(path_obj) -> str:
    """Convert path to project-relative string when possible for portable reports."""
    p = Path(path_obj)
    try:
        root = DATABASE_DIR.parent
        return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


class FineTuneService:
    def __init__(self, storage: StorageAdapter, person_name: str):
        self.storage = storage
        self.person_name = person_name
        self.cfg = FINE_TUNE_CONFIG

        self.model_adapter = ModelAdapter(
            scrfd_path=EMBEDDING_PATHS["SCRFD_MODEL"],
            arcface_path=EMBEDDING_PATHS["ARCFACE_MODEL"],
        )

        logger.info("%s FineTuneService initialized", INFO)
        logger.info("%s person=%s", METADATA, self.person_name)
        logger.info("%s SCRFD model: %s", METADATA, EMBEDDING_PATHS["SCRFD_MODEL"])
        logger.info("%s ArcFace model: %s", METADATA, EMBEDDING_PATHS["ARCFACE_MODEL"])
        logger.info(
            "%s cfg: epochs=%d batch=%d lr=%.6f wd=%.6f val_split=%.2f seed=%d",
            METADATA,
            self.cfg["EPOCHS"],
            self.cfg["BATCH_SIZE"],
            self.cfg["LEARNING_RATE"],
            self.cfg["WEIGHT_DECAY"],
            self.cfg["VAL_SPLIT"],
            self.cfg["SEED"],
        )

    def _load_replay_frames(self) -> List[Tuple[str, Path]]:
        replay_dir = self.cfg["REPLAY_ROOT_DIR"]
        items: List[Tuple[str, Path]] = []
        if not replay_dir.exists():
            return items

        excluded = set(self.cfg.get("REPLAY_EXCLUDE_DIRS", []))

        for person_dir in replay_dir.iterdir():
            if not person_dir.is_dir():
                continue
            if person_dir.name in excluded:
                continue
            if person_dir.name == self.person_name:
                continue
            for img in person_dir.rglob("*.jpg"):
                items.append((person_dir.name, img))
        return items

    def _load_current_person_frames(self) -> List[Tuple[str, Path]]:
        base = get_person_sanitized_dir(self.person_name)
        items: List[Tuple[str, Path]] = []
        for angle in ["frontal", "horizontal", "vertical"]:
            angle_dir = base / angle
            if not angle_dir.exists():
                continue
            for img in angle_dir.glob("*.jpg"):
                items.append((self.person_name, img))
        return items

    def _extract_embedding_from_image(self, image_path: Path) -> np.ndarray | None:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        faces = self.model_adapter.detect_faces(img, return_dataclass=False)
        if faces:
            best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
            aligned = self.model_adapter.align_face(img, best["landmarks"], target_size=(112, 112))
            if aligned is not None:
                emb = self.model_adapter.extract_embedding(aligned)
                return emb.astype(np.float32) if emb is not None else None

        resized = cv2.resize(img, (112, 112))
        emb = self.model_adapter.extract_embedding(resized)
        return emb.astype(np.float32) if emb is not None else None

    def _build_training_set(self) -> Tuple[np.ndarray, List[str]]:
        replay_items = self._load_replay_frames()
        current_items = self._load_current_person_frames()
        all_items = replay_items + current_items

        logger.info(f"{INFO} Fine-tune dataset build")
        logger.info(f"{METADATA} Replay frames: %d", len(replay_items))
        logger.info(f"{METADATA} Current person frames: %d", len(current_items))

        embeddings: List[np.ndarray] = []
        labels: List[str] = []

        for label, path in all_items:
            emb = self._extract_embedding_from_image(path)
            if emb is None:
                continue
            embeddings.append(emb)
            labels.append(label)

        if not embeddings:
            raise RuntimeError("No embeddings extracted for fine-tune dataset")

        x = np.stack(embeddings).astype(np.float32)
        logger.info(f"{SUCCESS} Training embeddings ready: shape=%s", x.shape)
        unique, counts = np.unique(np.array(labels), return_counts=True)
        class_dist = {str(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        logger.info(f"{METADATA} Class distribution: %s", class_dist)
        return x, labels

    def _save_plots(self, metrics: List[FineTuneEpochMetrics]) -> None:
        plots_dir = self.cfg["PLOTS_DIR"]
        plots_dir.mkdir(parents=True, exist_ok=True)

        epochs = [m.epoch for m in metrics]
        train_loss = [m.train_loss for m in metrics]
        val_loss = [m.val_loss for m in metrics]
        train_acc = [m.train_acc for m in metrics]
        val_acc = [m.val_acc for m in metrics]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Fine-tune Loss by Epoch")
        plt.legend()
        plt.grid(True, alpha=0.3)
        loss_path = plots_dir / "fine_tune_loss.png"
        plt.savefig(loss_path, dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_acc, label="train_acc")
        plt.plot(epochs, val_acc, label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Fine-tune Accuracy by Epoch")
        plt.legend()
        plt.grid(True, alpha=0.3)
        acc_path = plots_dir / "fine_tune_accuracy.png"
        plt.savefig(acc_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"{SUCCESS} Saved plots: %s, %s", loss_path, acc_path)

    def _compute_epoch_score(self, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> float:
        """Compute balanced checkpoint score (higher is better).

        Score components:
        - accuracy term: val_acc
        - loss term: 1 / (1 + val_loss)
        - balance penalty: |train_acc - val_acc|
        """
        w_acc = float(self.cfg.get("SCORE_WEIGHT_ACC", 0.5))
        w_loss = float(self.cfg.get("SCORE_WEIGHT_LOSS", 0.5))
        balance_penalty = float(self.cfg.get("SCORE_BALANCE_PENALTY", 0.2))

        loss_term = 1.0 / (1.0 + max(0.0, float(val_loss)))
        gap = abs(float(train_acc) - float(val_acc))
        score = w_acc * float(val_acc) + w_loss * loss_term - balance_penalty * gap
        return float(score)

    @staticmethod
    def _compute_per_class_val_acc(
        model: nn.Module,
        loader,
        idx_to_class: Dict[int, str],
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()
        correct: Dict[int, int] = {}
        total: Dict[int, int] = {}

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)

                for y, p in zip(yb.cpu().tolist(), preds.cpu().tolist()):
                    total[y] = total.get(y, 0) + 1
                    if y == p:
                        correct[y] = correct.get(y, 0) + 1

        result: Dict[str, float] = {}
        for idx, cnt in total.items():
            name = idx_to_class.get(idx, str(idx))
            result[name] = float(correct.get(idx, 0) / max(1, cnt))
        return result

    def _save_run_summary(
        self,
        report: FineTuneReport,
        labels: List[str],
        run_mode: str,
    ) -> Path:
        """Save JSON summary for later analysis/tuning."""
        unique, counts = np.unique(np.array(labels), return_counts=True)
        class_distribution = {str(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}

        summary = {
            "timestamp": datetime.now().isoformat(),
            "person_name": report.person_name,
            "run_mode": run_mode,
            "config": {
                "epochs": int(self.cfg["EPOCHS"]),
                "batch_size": int(self.cfg["BATCH_SIZE"]),
                "learning_rate": float(self.cfg["LEARNING_RATE"]),
                "weight_decay": float(self.cfg["WEIGHT_DECAY"]),
                "val_split": float(self.cfg["VAL_SPLIT"]),
                "seed": int(self.cfg["SEED"]),
                "device": str(self.cfg["DEVICE"]),
            },
            "model_paths": {
                "scrfd": _to_portable_path(EMBEDDING_PATHS["SCRFD_MODEL"]),
                "arcface": _to_portable_path(EMBEDDING_PATHS["ARCFACE_MODEL"]),
                "checkpoint": _to_portable_path(self.cfg["CHECKPOINT_PATH"]),
            },
            "result": {
                "best_val_acc": float(report.best_val_acc),
                "best_epoch": int(report.best_epoch),
                "best_score": float(report.best_score),
                "early_stopped": bool(report.early_stopped),
                "stopped_epoch": int(report.stopped_epoch),
                "epochs": int(report.epochs),
                "class_to_index": report.class_to_index,
                "class_distribution": class_distribution,
            },
            "epoch_metrics": [
                {
                    "epoch": int(m.epoch),
                    "train_loss": float(m.train_loss),
                    "train_acc": float(m.train_acc),
                    "val_loss": float(m.val_loss),
                    "val_acc": float(m.val_acc),
                    "score": float(m.score),
                    "per_class_val_acc": m.per_class_val_acc,
                }
                for m in report.metrics
            ],
        }

        summary_dir = Path(self.cfg["PLOTS_DIR"])
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"fine_tune_summary_{self.person_name}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"{SUCCESS} Saved fine-tune summary: %s", summary_path)
        return summary_path

    def _refresh_minio_npz_and_sqlite(self, class_to_index: Dict[str, int]) -> None:
        storage_key = IOT_MINIO_KEYS["DATABASE_NPZ"]
        merged_db: Dict[str, np.ndarray] = {}

        existing_bytes = self.storage.get(storage_key, use_cache=False)
        if existing_bytes:
            existing_npz = np.load(Path(".") / "__tmp__.npz", allow_pickle=False) if False else None
            existing_npz = np.load(__import__("io").BytesIO(existing_bytes), allow_pickle=False)
            for name in existing_npz.files:
                merged_db[name] = existing_npz[name].astype(np.float32)
        logger.info(f"{METADATA} Existing MinIO persons: %d", len(merged_db))

        # Recompute center vectors from replay + current data
        x, labels = self._build_training_set()
        for name in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == name]
            center = x[idx].mean(axis=0)
            norm = np.linalg.norm(center)
            if norm > 0:
                center = center / norm
            merged_db[name] = center.astype(np.float32)

        import io
        buff = io.BytesIO()
        np.savez_compressed(buff, **merged_db)
        if not self.storage.put(storage_key, buff.getvalue()):
            raise RuntimeError("Failed to upload merged recognition DB to MinIO")
        logger.info(f"{SUCCESS} Uploaded merged DB to MinIO key=%s persons=%d", storage_key, len(merged_db))

        sqlite = get_sqlite_manager()
        for name in sorted(set(labels)):
            person = sqlite.get_person_by_name(name)
            if person is None:
                sqlite.add_person(
                    name=name,
                    full_name=name,
                    vector_storage_key=storage_key,
                    vector_count=1,
                    metadata={"fine_tuned": True},
                )
            else:
                sqlite.update_person(
                    person["person_id"],
                    status="active",
                    vector_storage_key=storage_key,
                    vector_count=1,
                    metadata={"fine_tuned": True},
                )

        logger.info(f"{SUCCESS} Runtime sync complete (MinIO + SQLite)")
        logger.info(f"{METADATA} class_to_index used for deployment: %s", class_to_index)

    def _save_replay_samples(self) -> None:
        person_frame_root = get_person_sanitized_dir(self.person_name)
        replay_root = self.cfg["REPLAY_ROOT_DIR"] / self.person_name
        replay_root.mkdir(parents=True, exist_ok=True)

        keep_ratio = self.cfg["REPLAY_KEEP_RATIO"]
        min_keep = self.cfg["REPLAY_MIN_PER_ANGLE"]

        target_keep = int(self.cfg.get("REPLAY_SAMPLES_PER_ANGLE", min_keep))
        if bool(self.cfg.get("REPLAY_DYNAMIC_ENABLED", False)):
            sqlite = get_sqlite_manager()
            active_users = len(sqlite.list_persons(status="active"))
            base = float(self.cfg.get("REPLAY_DYNAMIC_BASE_PER_ANGLE", 50))
            slope_k = float(self.cfg.get("REPLAY_DYNAMIC_K", 2.0))
            dyn_min = int(self.cfg.get("REPLAY_DYNAMIC_MIN_PER_ANGLE", min_keep))
            dyn_max = int(self.cfg.get("REPLAY_DYNAMIC_MAX_PER_ANGLE", 80))
            linear = int(round(base - slope_k * active_users))
            target_keep = max(dyn_min, min(dyn_max, linear))
            logger.info(
                f"{METADATA} Dynamic replay target per angle: %d (base=%.1f K=%.3f users=%d)",
                target_keep,
                base,
                slope_k,
                active_users,
            )

        for angle in ["frontal", "horizontal", "vertical"]:
            src_dir = person_frame_root / angle
            if not src_dir.exists():
                continue
            files = sorted(src_dir.glob("*.jpg"))
            if not files:
                continue

            k = max(min_keep, int(len(files) * keep_ratio), target_keep)
            k = min(k, len(files))
            selected = random.sample(files, k)

            dst_dir = replay_root / angle
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in selected:
                shutil.copy2(f, dst_dir / f.name)

            logger.info(
                f"{METADATA} Replay angle=%s selected=%d/%d", angle, len(selected), len(files)
            )

        logger.info(f"{SUCCESS} Replay samples saved: %s", replay_root)

    def _cleanup_temp(self) -> None:
        from config.settings import get_person_temp_dir

        temp_dir = get_person_temp_dir(self.person_name)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"{SUCCESS} Temp cleaned: %s", temp_dir)

    def run(self) -> FineTuneReport:
        torch.manual_seed(self.cfg["SEED"])
        np.random.seed(self.cfg["SEED"])
        random.seed(self.cfg["SEED"])

        x, labels = self._build_training_set()
        class_to_idx = FineTuneAdapter.build_class_mapping(labels)
        y = FineTuneAdapter.encode_labels(labels, class_to_idx)
        logger.info(f"{METADATA} Classes mapped: %s", class_to_idx)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        ft = FineTuneAdapter(
            input_dim=x.shape[1],
            num_classes=len(class_to_idx),
            device=self.cfg["DEVICE"],
        )

        train_loader, val_loader = FineTuneAdapter.build_dataloaders(
            x,
            y,
            batch_size=self.cfg["BATCH_SIZE"],
            val_split=self.cfg["VAL_SPLIT"],
            seed=self.cfg["SEED"],
        )
        logger.info(
            f"{METADATA} Dataloaders | train_batches=%d val_batches=%d",
            len(train_loader),
            len(val_loader),
        )

        if len(class_to_idx) <= 1:
            logger.warning(
                f"{WARNING} Single-class dataset detected (%d class). "
                "Metrics can be inflated and not representative. Add replay/new users for meaningful evaluation.",
                len(class_to_idx),
            )

        idx_to_class = {v: k for k, v in class_to_idx.items()}

        criterion = nn.CrossEntropyLoss()
        class_count = len(class_to_idx)
        wd = float(self.cfg["WEIGHT_DECAY"])
        if class_count >= int(self.cfg.get("WEIGHT_DECAY_RELAX_CLASS_COUNT", 100)):
            wd = float(self.cfg.get("WEIGHT_DECAY_LARGE_CLASS", wd))

        optimizer = torch.optim.Adam(
            ft.model.parameters(),
            lr=self.cfg["LEARNING_RATE"],
            weight_decay=wd,
        )

        batch_epoch_size = int(self.cfg.get("BATCH_EPOCH_SIZE", 5))
        eta_min = float(self.cfg.get("MIN_LEARNING_RATE", 1e-4))
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, batch_epoch_size),
            eta_min=eta_min,
        )

        metrics: List[FineTuneEpochMetrics] = []
        best_val_acc = -1.0
        best_epoch = 0

        logger.info(f"{INFO} Starting fine-tune loop (CPU)")
        logger.info(f"{METADATA} epochs=%d batch_size=%d lr=%.6f", self.cfg["EPOCHS"], self.cfg["BATCH_SIZE"], self.cfg["LEARNING_RATE"])

        for epoch in range(1, self.cfg["EPOCHS"] + 1):
            train_loss, train_acc = ft.train_one_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = ft.eval_one_epoch(val_loader, criterion)

            metrics.append(
                FineTuneEpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                )
            )

            logger.info(
                f"{METADATA} [EPOCH] %03d | train_loss=%.6f train_acc=%.4f | val_loss=%.6f val_acc=%.4f",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                ft.save_checkpoint(self.cfg["CHECKPOINT_PATH"], class_to_idx)
                logger.info(
                    f"{SUCCESS} New best checkpoint at epoch=%d val_acc=%.4f path=%s",
                    epoch,
                    val_acc,
                    self.cfg["CHECKPOINT_PATH"],
                )

        self._save_plots(metrics)

        if best_val_acc < self.cfg["MIN_VAL_ACCURACY"]:
            logger.warning(f"{WARNING} Fine-tune val_acc below threshold: %.4f < %.4f", best_val_acc, self.cfg["MIN_VAL_ACCURACY"])

        self._refresh_minio_npz_and_sqlite(class_to_idx)
        self._save_replay_samples()
        self._cleanup_temp()

        report = FineTuneReport(
            person_name=self.person_name,
            epochs=self.cfg["EPOCHS"],
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            metrics=metrics,
            class_to_index=class_to_idx,
        )
        self._save_run_summary(report, labels, run_mode="full")

        logger.info(f"{SUCCESS} Fine-tune completed | best_val_acc=%.4f at epoch=%d", best_val_acc, best_epoch)
        return report

    def run_training_only(self) -> FineTuneReport:
        """Run fine-tune only (no upload/sync/cleanup).

        Caller decides whether to deploy/upload afterward.
        """
        torch.manual_seed(self.cfg["SEED"])
        np.random.seed(self.cfg["SEED"])
        random.seed(self.cfg["SEED"])

        x, labels = self._build_training_set()
        class_to_idx = FineTuneAdapter.build_class_mapping(labels)
        y = FineTuneAdapter.encode_labels(labels, class_to_idx)
        logger.info(f"{METADATA} Classes mapped: %s", class_to_idx)
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        ft = FineTuneAdapter(
            input_dim=x.shape[1],
            num_classes=len(class_to_idx),
            device=self.cfg["DEVICE"],
        )

        train_loader, val_loader = FineTuneAdapter.build_dataloaders(
            x,
            y,
            batch_size=self.cfg["BATCH_SIZE"],
            val_split=self.cfg["VAL_SPLIT"],
            seed=self.cfg["SEED"],
        )
        logger.info(
            f"{METADATA} Dataloaders | train_batches=%d val_batches=%d",
            len(train_loader),
            len(val_loader),
        )

        criterion = nn.CrossEntropyLoss()
        class_count = len(class_to_idx)
        wd = float(self.cfg["WEIGHT_DECAY"])
        if class_count >= int(self.cfg.get("WEIGHT_DECAY_RELAX_CLASS_COUNT", 100)):
            wd = float(self.cfg.get("WEIGHT_DECAY_LARGE_CLASS", wd))

        optimizer = torch.optim.Adam(
            ft.model.parameters(),
            lr=self.cfg["LEARNING_RATE"],
            weight_decay=wd,
        )

        batch_epoch_size = int(self.cfg.get("BATCH_EPOCH_SIZE", 5))
        eta_min = float(self.cfg.get("MIN_LEARNING_RATE", 1e-4))
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, batch_epoch_size),
            eta_min=eta_min,
        )

        metrics: List[FineTuneEpochMetrics] = []
        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_score = float("-inf")
        best_epoch = 0

        max_epochs = int(min(self.cfg.get("MAX_EPOCHS", self.cfg["EPOCHS"]), self.cfg["EPOCHS"]))
        patience_batches = int(self.cfg.get("EARLY_STOP_PATIENCE_BATCHES", 2))
        loss_plateau_delta = float(self.cfg.get("LOSS_PLATEAU_DELTA", 1e-3))
        loss_plateau_patience_batches = int(self.cfg.get("LOSS_PLATEAU_PATIENCE_BATCHES", 1))
        no_improve_batches = 0
        no_loss_improve_batches = 0
        early_stopped = False
        stopped_epoch = max_epochs
        last_batch_best_val_loss = float("inf")
        current_batch_best_val_loss = float("inf")

        logger.info(f"{INFO} Starting fine-tune loop (CPU) [training-only]")
        logger.info(
            f"{METADATA} smart-train cfg | max_epochs=%d batch_epoch=%d patience_batches=%d loss_plateau_delta=%.6f",
            max_epochs,
            batch_epoch_size,
            patience_batches,
            loss_plateau_delta,
        )
        logger.info(
            f"{METADATA} lr cycle cfg | lr_max=%.6f lr_min=%.6f",
            float(self.cfg["LEARNING_RATE"]),
            eta_min,
        )

        for epoch in range(1, max_epochs + 1):
            # Reset LR to max at the beginning of each batch cycle
            if (epoch - 1) % batch_epoch_size == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = float(self.cfg["LEARNING_RATE"])
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, batch_epoch_size),
                    eta_min=eta_min,
                )

            train_loss, train_acc = ft.train_one_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = ft.eval_one_epoch(val_loader, criterion)
            epoch_score = self._compute_epoch_score(train_loss, train_acc, val_loss, val_acc)
            per_class_val_acc = self._compute_per_class_val_acc(
                ft.model,
                val_loader,
                {v: k for k, v in class_to_idx.items()},
                ft.device,
            )

            metrics.append(
                FineTuneEpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    score=epoch_score,
                    per_class_val_acc=per_class_val_acc,
                )
            )

            logger.info(
                f"{METADATA} [EPOCH] %03d | lr=%.6f | train_loss=%.6f train_acc=%.4f | val_loss=%.6f val_acc=%.4f | score=%.6f",
                epoch,
                optimizer.param_groups[0]["lr"],
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                epoch_score,
            )
            if per_class_val_acc:
                logger.info(f"{METADATA} [EPOCH] %03d per-class val acc: %s", epoch, per_class_val_acc)

            improved = False
            if epoch_score > best_score:
                improved = True
                best_score = epoch_score
                best_val_acc = max(best_val_acc, val_acc)
                best_val_loss = min(best_val_loss, val_loss)
                best_epoch = epoch
                ft.save_checkpoint(self.cfg["CHECKPOINT_PATH"], class_to_idx)
                logger.info(
                    f"{SUCCESS} New best checkpoint at epoch=%d score=%.6f val_acc=%.4f val_loss=%.6f path=%s",
                    epoch,
                    epoch_score,
                    val_acc,
                    val_loss,
                    self.cfg["CHECKPOINT_PATH"],
                )
            else:
                best_val_acc = val_acc
                best_val_loss = min(best_val_loss, val_loss)

            current_batch_best_val_loss = min(current_batch_best_val_loss, float(val_loss))

            scheduler.step()

            if epoch % batch_epoch_size == 0:
                batch_id = epoch // batch_epoch_size
                if improved:
                    no_improve_batches = 0
                    logger.info(
                        f"{METADATA} [BATCH %02d] improved | checkpoint_epoch=%d score=%.6f",
                        batch_id,
                        best_epoch,
                        best_score,
                    )
                else:
                    no_improve_batches += 1
                    logger.info(
                        f"{WARNING} [BATCH %02d] no improvement | stale_batches=%d/%d",
                        batch_id,
                        no_improve_batches,
                        patience_batches,
                    )

                # Loss plateau detection by batch
                if last_batch_best_val_loss < float("inf"):
                    loss_gain = last_batch_best_val_loss - current_batch_best_val_loss
                    if loss_gain < loss_plateau_delta:
                        no_loss_improve_batches += 1
                        logger.info(
                            f"{WARNING} [BATCH %02d] loss plateau | gain=%.6f < delta=%.6f | stale_loss_batches=%d/%d",
                            batch_id,
                            loss_gain,
                            loss_plateau_delta,
                            no_loss_improve_batches,
                            loss_plateau_patience_batches,
                        )
                    else:
                        no_loss_improve_batches = 0

                last_batch_best_val_loss = current_batch_best_val_loss
                current_batch_best_val_loss = float("inf")

                if no_improve_batches >= patience_batches:
                    early_stopped = True
                    stopped_epoch = epoch
                    logger.info(
                        f"{SUCCESS} Early stopping triggered at epoch=%d (no improvement for %d batches)",
                        epoch,
                        no_improve_batches,
                    )
                    break

                if no_loss_improve_batches >= loss_plateau_patience_batches:
                    early_stopped = True
                    stopped_epoch = epoch
                    logger.info(
                        f"{SUCCESS} Early stopping triggered at epoch=%d due to loss plateau",
                        epoch,
                    )
                    break

        self._save_plots(metrics)

        if best_val_acc < self.cfg["MIN_VAL_ACCURACY"]:
            logger.warning(f"{WARNING} Fine-tune val_acc below threshold: %.4f < %.4f", best_val_acc, self.cfg["MIN_VAL_ACCURACY"])

        report = FineTuneReport(
            person_name=self.person_name,
            epochs=len(metrics),
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            best_score=best_score if best_score != float("-inf") else 0.0,
            early_stopped=early_stopped,
            stopped_epoch=stopped_epoch,
            metrics=metrics,
            class_to_index=class_to_idx,
        )
        self._save_run_summary(report, labels, run_mode="training_only")
        logger.info(f"{SUCCESS} Fine-tune training-only complete | best_val_acc=%.4f at epoch=%d", best_val_acc, best_epoch)
        return report

    def deploy_after_training(self, class_to_index: Dict[str, int], cleanup_temp: bool = True) -> None:
        """Deploy artifacts after user confirmation: MinIO + SQLite + replay + cleanup."""
        self._refresh_minio_npz_and_sqlite(class_to_index)
        self._save_replay_samples()
        if cleanup_temp:
            self._cleanup_temp()


__all__ = ["FineTuneService"]
