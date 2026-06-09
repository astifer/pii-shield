"""Entity-level metrics for the HF Trainer and standalone evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .labels import id2label

Entity = tuple[int, int, str]


def entities_from_tags(tags: list[str]) -> set[tuple[str, int, int]]:
    """Extract ``(type, start_token, end_token)`` spans from a BIO tag sequence."""
    entities: set[tuple[str, int, int]] = set()
    cur_type: str | None = None
    cur_start: int | None = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur_type is not None:
                entities.add((cur_type, cur_start, i))
                cur_type, cur_start = None, None
            continue
        prefix, typ = tag.split("-", 1)
        if prefix == "B":
            if cur_type is not None:
                entities.add((cur_type, cur_start, i))
            cur_type, cur_start = typ, i
        elif prefix == "I":
            if cur_type != typ:
                if cur_type is not None:
                    entities.add((cur_type, cur_start, i))
                cur_type, cur_start = typ, i
    if cur_type is not None:
        entities.add((cur_type, cur_start, len(tags)))
    return entities


def compute_metrics(eval_pred) -> dict[str, float]:
    """HF Trainer ``compute_metrics``: token accuracy + entity-level P/R/F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)

    total = (labels != -100).sum()
    correct = (preds == labels).astype(int)[labels != -100].sum()
    token_accuracy = correct / total if total else 0.0

    tp = fp = fn = 0
    for pred_seq, label_seq in zip(preds, labels):
        t_seq, p_seq = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            t_seq.append(id2label[l])
            p_seq.append(id2label[p])
        t_set = entities_from_tags(t_seq)
        p_set = entities_from_tags(p_seq)
        tp += len(t_set & p_set)
        fp += len(p_set - t_set)
        fn += len(t_set - p_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": token_accuracy}


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def span_prf1(gold: list[list[Entity]], pred: list[list[Entity]]) -> dict[str, Any]:
    """Micro and per-label P/R/F1 over exact ``(start, end, label)`` char spans.

    ``gold`` and ``pred`` are aligned lists, one entry per example.
    """
    tp = fp = fn = 0
    per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # label -> [tp, fp, fn]

    for gold_ents, pred_ents in zip(gold, pred):
        g = {(int(s), int(e), str(l)) for s, e, l in gold_ents}
        p = {(int(s), int(e), str(l)) for s, e, l in pred_ents}
        for ent in g & p:
            per_label[ent[2]][0] += 1
        for ent in p - g:
            per_label[ent[2]][1] += 1
        for ent in g - p:
            per_label[ent[2]][2] += 1
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)

    precision, recall, f1 = _prf1(tp, fp, fn)
    labels = {}
    for label, (lt, lfp, lfn) in sorted(per_label.items()):
        lp, lr, lf = _prf1(lt, lfp, lfn)
        labels[label] = {
            "precision": lp, "recall": lr, "f1": lf,
            "tp": lt, "fp": lfp, "fn": lfn, "support": lt + lfn,
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "per_label": labels,
    }
