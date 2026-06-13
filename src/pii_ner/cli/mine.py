"""Mine uncertainty from saved fold checkpoints (no retraining).

Reconstructs the OOF KFold split and writes hard_examples_train.csv,
hard_examples_private.csv and train_private.csv (confident pseudo-labels).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pii_ner import config
from pii_ner.data import load_test, load_train
from pii_ner.inference import load_checkpoint
from pii_ner.mining import mine_private, mine_train_oof


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uncertainty mining from saved checkpoints")
    p.add_argument("--model-dirs", nargs="+", required=True, help="Fold checkpoint dirs, in fold order")
    p.add_argument("--folds", type=int, default=None, help="Folds used in training (default: #dirs)")
    p.add_argument("--train-file", default=str(config.TRAIN_FILE))
    p.add_argument("--test-file", default=str(config.TEST_FILE))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = config.get_device()
    n_folds = args.folds or len(args.model_dirs)
    print(f"Device: {device} | folds={n_folds}")

    pairs = [load_checkpoint(Path(d), device) for d in args.model_dirs]

    all_data = load_train(args.train_file)
    test_texts = load_test(args.test_file)
    print(f"Train: {len(all_data)} | Private: {len(test_texts)}")

    hard_train = mine_train_oof(all_data, pairs, n_folds)
    pd.DataFrame(hard_train).to_csv("hard_examples_train.csv", index=False)
    print(f"Saved hard_examples_train.csv ({len(hard_train)} examples)")

    hard_private, confident = mine_private(test_texts, pairs)
    pd.DataFrame(hard_private).to_csv("hard_examples_private.csv", index=False)
    pd.DataFrame(
        [{"text": x["text"], "target": str(x["entities"])} for x in confident]
    ).to_csv("train_private.csv", index=False)
    print(f"Saved hard_examples_private.csv ({len(hard_private)} hard) and "
          f"train_private.csv ({len(confident)} confident)")


if __name__ == "__main__":
    main()
