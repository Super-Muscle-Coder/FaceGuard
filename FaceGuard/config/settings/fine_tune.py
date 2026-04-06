"""Fine-tune configuration for incremental ArcFace-style training workflow.

This config is dedicated to Phase 5 fine-tune mode (CPU-friendly).
It focuses on classifier-head training over ArcFace embeddings with replay.
"""

from .path import ROOT_DIR, DATABASE_DIR, DATA_DIR

FINE_TUNE_CONFIG = {
    # Runtime
    "ENABLED": True,
    "DEVICE": "cpu",

    # Optimizer / training
    "EPOCHS": 100,
    "MAX_EPOCHS": 100,
    "BATCH_EPOCH_SIZE": 5,
    "EARLY_STOP_PATIENCE_BATCHES": 2,
    "LOSS_PLATEAU_DELTA": 1e-3,
    "LOSS_PLATEAU_PATIENCE_BATCHES": 1,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-3,
    "MIN_LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "WEIGHT_DECAY_LARGE_CLASS": 5e-5,
    "WEIGHT_DECAY_RELAX_CLASS_COUNT": 100,
    "VAL_SPLIT": 0.2,
    "SEED": 42,

    # Smart checkpoint score (balanced loss + acc)
    "SCORE_WEIGHT_ACC": 0.5,
    "SCORE_WEIGHT_LOSS": 0.5,
    "SCORE_BALANCE_PENALTY": 0.2,

    # Replay strategy (per identity, per angle)
    "REPLAY_KEEP_RATIO": 0.05,
    "REPLAY_MIN_PER_ANGLE": 20,
    "REPLAY_SAMPLES_PER_ANGLE": 20,
    # Dynamic replay policy: n = base - K*i (i = active users), clamped by min/max
    "REPLAY_DYNAMIC_ENABLED": True,
    "REPLAY_DYNAMIC_BASE_PER_ANGLE": 50,
    "REPLAY_DYNAMIC_K": 2.0,
    "REPLAY_DYNAMIC_MIN_PER_ANGLE": 40,
    "REPLAY_DYNAMIC_MAX_PER_ANGLE": 80,

    # Quality gate for accepting fine-tune result
    "MIN_VAL_ACCURACY": 0.70,

    # Artifacts
    "CHECKPOINT_PATH": DATABASE_DIR / "fine_tune_head.pt",
    "PLOTS_DIR": ROOT_DIR / "training_plots",
    # Replay buffer storage policy:
    # save each person at data/{person_name}/(frontal|horizontal|vertical)
    "REPLAY_ROOT_DIR": DATA_DIR,
    "REPLAY_EXCLUDE_DIRS": ["temp", ".cache", "replay_bank"],
}

__all__ = ["FINE_TUNE_CONFIG"]
