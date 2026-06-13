"""Uncertainty mining: surface hard examples and confident pseudo-labels.

train (OOF): each example is scored by the fold model that held it out, so the
signal is leakage-free. private (blended): all fold models are averaged.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import KFold
from tqdm import tqdm

from .config import CONFIDENT_THRESH, SEED, UNCERTAIN_THRESH
from .inference import (
    ModelTokenizer,
    blend_probs,
    compute_uncertainty,
    extract_entities,
    get_token_probs,
)

Example = dict[str, Any]


def fold_assignment(n_examples: int, n_folds: int, seed: int = SEED) -> dict[int, int]:
    """Map example index -> held-out fold index, matching the training KFold."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    idx_to_fold: dict[int, int] = {}
    for fold_i, (_, val_idx) in enumerate(kf.split(range(n_examples))):
        for i in val_idx:
            idx_to_fold[int(i)] = fold_i
    return idx_to_fold


def mine_train_oof(
    all_data: list[Example],
    fold_pairs: list[ModelTokenizer],
    n_folds: int,
    uncertain_thresh: float = UNCERTAIN_THRESH,
) -> list[dict]:
    """Flag train examples mispredicted or high-uncertainty by their OOF model."""
    idx_to_fold = fold_assignment(len(all_data), n_folds)
    hard: list[dict] = []
    for ex_i, item in enumerate(tqdm(all_data, desc="Mining train (OOF)")):
        fold_i = idx_to_fold.get(ex_i)
        if fold_i is None or fold_i >= len(fold_pairs):
            continue
        model, tokenizer = fold_pairs[fold_i]
        offsets, probs = get_token_probs(item["text"], model, tokenizer)
        uncertainty = compute_uncertainty(offsets, probs)
        predicted = set(extract_entities(offsets, probs))
        true = {tuple(e) for e in item["entities"]}
        is_incorrect = predicted != true
        if uncertainty > uncertain_thresh or is_incorrect:
            hard.append({
                "text": item["text"],
                "target": str(item["entities"]),
                "predicted": str(list(predicted)),
                "uncertainty_score": uncertainty,
                "is_incorrect": is_incorrect,
            })
    return hard


def mine_private(
    test_texts: list[Example],
    fold_pairs: list[ModelTokenizer],
    uncertain_thresh: float = UNCERTAIN_THRESH,
    confident_thresh: float = CONFIDENT_THRESH,
) -> tuple[list[dict], list[dict]]:
    """Split private examples into hard (high uncertainty) and confident (low)."""
    hard: list[dict] = []
    confident: list[dict] = []
    for item in tqdm(test_texts, desc="Mining private (blended)"):
        offsets, avg_probs = blend_probs(item["text"], fold_pairs)
        uncertainty = compute_uncertainty(offsets, avg_probs)
        entities = extract_entities(offsets, avg_probs)
        if uncertainty > uncertain_thresh:
            hard.append({
                "id": item["id"],
                "text": item["text"],
                "predicted": str(entities),
                "uncertainty_score": uncertainty,
            })
        elif uncertainty < confident_thresh:
            confident.append({
                "id": item["id"],
                "text": item["text"],
                "entities": entities,
                "uncertainty_score": uncertainty,
            })
    return hard, confident
