"""Fine-tune adapter (PyTorch) for incremental classifier-head training.

Adapter responsibility:
- Build CPU-friendly dataloaders from embeddings + labels
- Train/evaluate a lightweight linear classification head
- Save/load checkpoint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger(__name__)


@dataclass
class FineTuneBatch:
    embeddings: np.ndarray
    labels: np.ndarray


class FineTuneAdapter:
    def __init__(self, input_dim: int, num_classes: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = nn.Linear(input_dim, num_classes).to(self.device)

    @staticmethod
    def build_class_mapping(labels: List[str]) -> Dict[str, int]:
        classes = sorted(set(labels))
        return {name: idx for idx, name in enumerate(classes)}

    @staticmethod
    def encode_labels(labels: List[str], class_to_idx: Dict[str, int]) -> np.ndarray:
        return np.array([class_to_idx[l] for l in labels], dtype=np.int64)

    @staticmethod
    def build_dataloaders(
        embeddings: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        val_split: float,
        seed: int,
    ) -> Tuple[DataLoader, DataLoader]:
        x = torch.tensor(embeddings, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(x, y)

        val_size = max(1, int(len(dataset) * val_split))
        train_size = max(1, len(dataset) - val_size)
        if train_size + val_size > len(dataset):
            val_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        )

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            optimizer.zero_grad()
            logits = self.model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == yb).sum().item())
            total += xb.size(0)

        return total_loss / max(1, total), total_correct / max(1, total)

    def eval_one_epoch(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                logits = self.model(xb)
                loss = criterion(logits, yb)

                total_loss += float(loss.item()) * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += int((preds == yb).sum().item())
                total += xb.size(0)

        return total_loss / max(1, total), total_correct / max(1, total)

    def save_checkpoint(self, checkpoint_path: Path, class_to_index: Dict[str, int]) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "class_to_index": class_to_index,
            },
            checkpoint_path,
        )
        logger.info("Saved fine-tune checkpoint: %s", checkpoint_path)


__all__ = ["FineTuneAdapter", "FineTuneBatch"]
