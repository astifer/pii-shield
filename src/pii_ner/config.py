"""Shared paths, decoding thresholds and training hyperparameters."""

from __future__ import annotations

from pathlib import Path

import torch

DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "train_dataset.tsv"
TEST_FILE = DATA_DIR / "private_test_dataset.csv"
DEFAULT_SUBMISSION = "submission.csv"

MODEL_RUBERT = "DeepPavlov/rubert-base-cased"
MODEL_CONV = "Data-Lab/rubert-base-cased-conversational_ner-v1"

MAX_LENGTH = 512
THRESHOLD = 0.6  # min softmax prob to accept a non-"O" label

N_FOLDS = 3
BATCH_SIZE = 256
GRAD_ACCUM = 1
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
LOGGING_STEPS = 50
EVAL_STEPS = 300
VAL_SAMPLE = 600
SEED = 42

# Uncertainty = 1 - geometric_mean(max_prob) over real tokens.
UNCERTAIN_THRESH = 0.15  # above -> hard example
CONFIDENT_THRESH = 0.12  # below -> confident enough to pseudo-label


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
