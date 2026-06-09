"""Train token-classification models with k-fold CV and optional uncertainty mining.

Fold checkpoints go to ``ner_model_<tag>_fold{N}/`` (``tag`` = model name's last
path component). Existing checkpoints are loaded instead of retrained, so this is
resumable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold
from transformers import AutoTokenizer

from pii_ner import config
from pii_ner.data import load_test, load_train
from pii_ner.inference import load_checkpoint
from pii_ner.mining import mine_private, mine_train_oof
from pii_ner.training import train_augmented, train_fold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NER models with k-fold CV")
    p.add_argument("--model", default=config.MODEL_RUBERT, help="HuggingFace model name")
    p.add_argument("--folds", type=int, default=config.N_FOLDS, help="Number of CV folds")
    p.add_argument("--train-file", default=str(config.TRAIN_FILE))
    p.add_argument("--test-file", default=str(config.TEST_FILE))
    p.add_argument("--mine", action="store_true",
                   help="After training, mine uncertainty and train an augmented model")
    p.add_argument("--ema", action="store_true", help="Use EMA-smoothed weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.enable_tf32()
    device = config.get_device()
    print(f"Device: {device} | Model: {args.model} | Folds: {args.folds}")

    model_tag = args.model.split("/")[-1]
    all_data = load_train(args.train_file)
    print(f"Loaded {len(all_data)} train examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=config.SEED)
    fold_pairs = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(all_data)))):
        save_path = Path(f"ner_model_{model_tag}_fold{fold + 1}")
        if (save_path / "config.json").exists():
            print(f"\nFold {fold + 1}: loading existing checkpoint {save_path}")
            model, _ = load_checkpoint(save_path, device)
        else:
            model = train_fold(
                fold,
                [all_data[i] for i in train_idx],
                [all_data[i] for i in val_idx],
                tokenizer,
                model_name=args.model,
                output_dir=str(save_path),
                use_ema=args.ema,
            )
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved fold {fold + 1} -> {save_path}")
            model = model.to(device).eval()
        fold_pairs.append((model, tokenizer))

    if not args.mine:
        print("\nDone (folds only). Run `pii-blend` to produce a submission.")
        return

    print("\nMining hard train examples (OOF)...")
    hard_train = mine_train_oof(all_data, fold_pairs, args.folds)
    pd.DataFrame(hard_train).to_csv("hard_examples_train.csv", index=False)
    print(f"Saved hard_examples_train.csv ({len(hard_train)} examples)")

    print("\nMining private examples (blended)...")
    test_texts = load_test(args.test_file)
    hard_private, confident = mine_private(test_texts, fold_pairs)
    pd.DataFrame(hard_private).to_csv("hard_examples_private.csv", index=False)
    pd.DataFrame(
        [{"text": x["text"], "target": str(x["entities"])} for x in confident]
    ).to_csv("train_private.csv", index=False)
    print(f"Saved hard_examples_private.csv ({len(hard_private)} hard) and "
          f"train_private.csv ({len(confident)} confident pseudo-labels)")

    aug_dir = f"ner_model_{model_tag}_augmented"
    aug_model = train_augmented(
        all_data, confident, tokenizer, model_name=args.model, output_dir=aug_dir, use_ema=args.ema
    )
    aug_model.save_pretrained(aug_dir)
    tokenizer.save_pretrained(aug_dir)
    print(f"Saved augmented model -> {aug_dir}")
    print("\nDone. Run `pii-blend` over the fold + augmented dirs to produce a submission.")


if __name__ == "__main__":
    main()
