"""Evaluate checkpoints on labeled data with span-level precision/recall/F1.

Blend eval (default) averages the given checkpoints; ``--oof`` scores each example
by the fold model that held it out (one checkpoint per fold, in order). ``--postprocess``
scores the predictions after submission postprocessing.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pii_ner import config
from pii_ner.data import load_train
from pii_ner.inference import blend_probs, extract_entities, get_token_probs, load_checkpoint
from pii_ner.metrics import span_prf1
from pii_ner.mining import fold_assignment
from pii_ner.postprocessing import PostprocessConfig, postprocess_ner_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NER checkpoints (span-level P/R/F1)")
    p.add_argument("--model-dirs", nargs="+", required=True, help="Checkpoint dir(s)")
    p.add_argument("--data", default=str(config.TRAIN_FILE), help="Labeled TSV to evaluate on")
    p.add_argument("--oof", action="store_true",
                   help="Score each example by its held-out fold model")
    p.add_argument("--folds", type=int, default=None,
                   help="Folds used in training (default: #model-dirs); only for --oof")
    p.add_argument("--threshold", type=float, default=config.THRESHOLD)
    p.add_argument("--postprocess", action="store_true",
                   help="Apply submission postprocessing before scoring")
    p.add_argument("--per-label", action="store_true", help="Print per-label breakdown")
    p.add_argument("--output", default=None, help="Optional CSV path for per-label metrics")
    return p.parse_args()


def _predict_oof(data, pairs, n_folds, threshold):
    idx_to_fold = fold_assignment(len(data), n_folds)
    preds = []
    for ex_i, item in enumerate(tqdm(data, desc="Predicting (OOF)")):
        fold_i = idx_to_fold.get(ex_i)
        if fold_i is None or fold_i >= len(pairs):
            preds.append([])
            continue
        model, tok = pairs[fold_i]
        off, probs = get_token_probs(item["text"], model, tok)
        preds.append(extract_entities(off, probs, threshold))
    return preds


def _predict_blend(data, pairs, threshold):
    preds = []
    for item in tqdm(data, desc="Predicting (blend)"):
        offsets, avg = blend_probs(item["text"], pairs)
        preds.append(extract_entities(offsets, avg, threshold))
    return preds


def _apply_postprocess(data, preds):
    ids = list(range(len(data)))
    preds_df = pd.DataFrame({"id": ids, "Prediction": [str(p) for p in preds]})
    texts_df = pd.DataFrame({"id": ids, "text": [d["text"] for d in data]})
    result_df, _ = postprocess_ner_predictions(preds_df, texts_df, cfg=PostprocessConfig())
    out = []
    for v in result_df["Prediction"]:
        try:
            out.append([tuple(e) for e in ast.literal_eval(v)])
        except Exception:
            out.append([])
    return out


def main() -> None:
    args = parse_args()
    device = config.get_device()
    print(f"Device: {device}")

    pairs = [load_checkpoint(Path(d), device) for d in args.model_dirs]
    data = load_train(args.data)
    gold = [item["entities"] for item in data]
    print(f"Evaluating on {len(data)} examples from {args.data}")

    if args.oof:
        n_folds = args.folds or len(pairs)
        preds = _predict_oof(data, pairs, n_folds, args.threshold)
    else:
        preds = _predict_blend(data, pairs, args.threshold)

    if args.postprocess:
        preds = _apply_postprocess(data, preds)

    report = span_prf1(gold, preds)

    mode = "OOF" if args.oof else "blend"
    pp = " +postprocess" if args.postprocess else ""
    print(f"\n=== Span-level metrics ({mode}{pp}, threshold={args.threshold}) ===")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall:    {report['recall']:.4f}")
    print(f"F1:        {report['f1']:.4f}")
    print(f"TP/FP/FN:  {report['tp']}/{report['fp']}/{report['fn']}  (support={report['support']})")

    if args.per_label or args.output:
        rows = [
            {"label": label, **{k: m[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "support")}}
            for label, m in report["per_label"].items()
        ]
        label_df = pd.DataFrame(rows).sort_values("support", ascending=False)
        if args.per_label:
            print("\nPer-label:")
            with pd.option_context("display.max_rows", None, "display.width", 200):
                print(label_df.to_string(index=False))
        if args.output:
            label_df.to_csv(args.output, index=False)
            print(f"\nSaved per-label metrics -> {args.output}")


if __name__ == "__main__":
    main()
