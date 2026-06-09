"""Model inference, probability blending and BIO decoding.

Models are ensembled by averaging their raw softmax arrays, so every caller must
decode probabilities through these functions rather than reimplementing them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .config import MAX_LENGTH, THRESHOLD, get_device
from .labels import id2label

Offsets = list[tuple[int, int]]
Entity = tuple[int, int, str]
ModelTokenizer = tuple[Any, Any]


def load_checkpoint(model_dir: str | Path, device: str | None = None) -> ModelTokenizer:
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    return model, tokenizer


def get_token_probs(text: str, model, tokenizer, max_length: int = MAX_LENGTH) -> tuple[Offsets, np.ndarray]:
    """Return ``(offset_mapping, probs)`` where ``probs`` is ``(seq_len, n_labels)``."""
    model.eval()
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)[0].cpu().numpy()
    return offsets, probs


def extract_entities(offsets: Offsets, probs: np.ndarray, threshold: float = THRESHOLD) -> list[Entity]:
    """Decode BIO entities from a per-token probability array.

    A token is "O" when its top probability is below ``threshold``; entities are
    emitted as ``(start_char, end_char, label)`` spans.
    """
    entities: list[Entity] = []
    current: Entity | None = None
    for (start, end), token_probs in zip(offsets, probs):
        if start == 0 and end == 0:  # special token
            if current is not None:
                entities.append(current)
                current = None
            continue

        pred_id = int(np.argmax(token_probs))
        label = "O" if float(token_probs[pred_id]) < threshold else id2label[pred_id]

        if label.startswith("B-"):
            if current is not None:
                entities.append(current)
            current = (start, end, label[2:])
        elif label.startswith("I-"):
            typ = label[2:]
            if current is not None and current[2] == typ:
                current = (current[0], end, typ)
            else:
                if current is not None:
                    entities.append(current)
                current = (start, end, typ)
        else:
            if current is not None:
                entities.append(current)
                current = None

    if current is not None:
        entities.append(current)
    return entities


def compute_uncertainty(offsets: Offsets, probs: np.ndarray) -> float:
    """Return ``1 - geometric_mean(max_prob)`` over real tokens (higher = less confident)."""
    log_max = [
        float(np.log(np.clip(p.max(), 1e-9, 1.0)))
        for (s, e), p in zip(offsets, probs)
        if not (s == 0 and e == 0)
    ]
    return 1.0 - float(np.exp(np.mean(log_max))) if log_max else 1.0


def blend_probs(text: str, pairs: list[ModelTokenizer]) -> tuple[Offsets, np.ndarray]:
    """Average softmax probabilities across all ``(model, tokenizer)`` pairs."""
    avg: np.ndarray | None = None
    offsets: Offsets | None = None
    for model, tokenizer in pairs:
        off, probs = get_token_probs(text, model, tokenizer)
        if avg is None:
            avg, offsets = probs.copy(), off
        else:
            avg += probs
    assert avg is not None and offsets is not None, "blend_probs requires >=1 model"
    return offsets, avg / len(pairs)


def predict_entities(text: str, pairs: list[ModelTokenizer], threshold: float = THRESHOLD) -> list[Entity]:
    offsets, probs = blend_probs(text, pairs)
    return extract_entities(offsets, probs, threshold)
