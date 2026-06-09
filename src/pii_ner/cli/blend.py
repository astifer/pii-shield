"""Blend saved checkpoints into a submission.

Averages softmax probabilities across all given checkpoints, decodes BIO entities,
applies postprocessing, and writes ``submission.csv`` plus a per-model
``blended_private.csv`` (predictions + uncertainty for inspection).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pii_ner import config
from pii_ner.data import load_test
from pii_ner.inference import (
    blend_probs,
    compute_uncertainty,
    extract_entities,
    get_token_probs,
    load_checkpoint,
)
from pii_ner.postprocessing import PostprocessConfig, postprocess_ner_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend NER checkpoints -> submission")
    p.add_argument("--model-dirs", nargs="+", required=True, help="Saved checkpoint dirs")
    p.add_argument("--test-file", default=str(config.TEST_FILE))
    p.add_argument("--output", default=config.DEFAULT_SUBMISSION)
    p.add_argument("--detail-output", default="blended_private.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = config.get_device()
    print(f"Device: {device}")

    model_dirs = [Path(d) for d in args.model_dirs]
    names = [d.name for d in model_dirs]
    print(f"Loading {len(model_dirs)} models...")
    pairs = [load_checkpoint(d, device) for d in model_dirs]

    test_texts = load_test(args.test_file)
    print(f"Loaded {len(test_texts)} test examples")

    blended: list = []
    per_model_preds: list[list[str]] = [[] for _ in pairs]
    per_model_unc: list[list[float]] = [[] for _ in pairs]

    for item in tqdm(test_texts, desc="Blending"):
        for i, (model, tok) in enumerate(pairs):
            off, probs = get_token_probs(item["text"], model, tok)
            per_model_preds[i].append(str(extract_entities(off, probs)))
            per_model_unc[i].append(compute_uncertainty(off, probs))
        offsets, avg = blend_probs(item["text"], pairs)
        blended.append(extract_entities(offsets, avg))

    detail = pd.DataFrame({"id": [int(t["id"]) for t in test_texts]})
    for i, name in enumerate(names):
        detail[f"pred_{name}"] = per_model_preds[i]
        detail[f"uncertainty_{name}"] = per_model_unc[i]
    detail.to_csv(args.detail_output, index=False)
    print(f"Saved {args.detail_output} ({len(detail)} rows, {len(names)} models)")

    preds_df = pd.DataFrame({
        "id": [int(t["id"]) for t in test_texts],
        "Prediction": [str(ents) for ents in blended],
    })
    texts_df = pd.DataFrame({
        "id": [int(t["id"]) for t in test_texts],
        "text": [t["text"] for t in test_texts],
    })

    print("Applying postprocessing...")
    result_df, report = postprocess_ner_predictions(preds_df, texts_df, cfg=PostprocessConfig())
    print("Postprocessing counts:", report["counts"])

    result_df.to_csv(args.output, index=False)
    print(f"\nSaved {args.output} ({len(result_df)} rows)")
    print(result_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
