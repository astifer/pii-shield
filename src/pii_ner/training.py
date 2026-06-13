"""Model training: per-fold fine-tuning and pseudo-label-augmented retraining."""

from __future__ import annotations

import random
from typing import Any

from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerControl,
    TrainerState,
)

from . import config
from .data import NERDataset
from .ema import EMACallback
from .labels import NUM_LABELS, id2label, label2id
from .metrics import compute_metrics

Example = dict[str, Any]


class MetricsLoggingCallback(TrainerCallback):
    """One-line logging of train/eval metrics."""

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        parts = [f"Step {state.global_step}"]
        for key, fmt in [
            ("loss", "Loss: {:.4f}"),
            ("learning_rate", "LR: {:.2e}"),
            ("eval_loss", "Eval Loss: {:.4f}"),
            ("eval_f1", "Eval F1: {:.4f}"),
            ("eval_precision", "Eval P: {:.4f}"),
            ("eval_recall", "Eval R: {:.4f}"),
        ]:
            if key in logs:
                parts.append(fmt.format(logs[key]))
        print(" | ".join(parts))


def _build_training_args(output_dir: str, report_to: str = "none") -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        label_smoothing_factor=config.LABEL_SMOOTHING,
        report_to=report_to,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )


def _new_model(model_name: str):
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def _trainer(model, args, train_ds, val_ds, use_ema: bool) -> Trainer:
    callbacks: list[TrainerCallback] = [MetricsLoggingCallback()]
    if use_ema:
        callbacks.append(EMACallback())
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )


def train_fold(
    fold: int,
    train_data: list[Example],
    val_data: list[Example],
    tokenizer,
    model_name: str,
    output_dir: str,
    use_ema: bool = False,
) -> Any:
    """Fine-tune one CV fold and return the trained model."""
    print(f"\n{'=' * 50}\nFOLD {fold + 1}  [{model_name}]\n{'=' * 50}")
    train_ds = NERDataset(train_data, tokenizer)

    val_sample = random.sample(val_data, min(config.VAL_SAMPLE, len(val_data)))
    val_ds = NERDataset(val_sample, tokenizer)

    model = _new_model(model_name)
    trainer = _trainer(model, _build_training_args(output_dir), train_ds, val_ds, use_ema)
    trainer.train()

    train_sample = random.sample(train_data, min(config.VAL_SAMPLE, len(train_data)))
    train_metrics = trainer.evaluate(NERDataset(train_sample, tokenizer))
    print(
        f"[fold{fold + 1}] Train F1: {train_metrics.get('eval_f1', 0):.4f} | "
        f"P: {train_metrics.get('eval_precision', 0):.4f} | "
        f"R: {train_metrics.get('eval_recall', 0):.4f}"
    )
    return model


def train_augmented(
    all_data: list[Example],
    pseudo_data: list[Example],
    tokenizer,
    model_name: str,
    output_dir: str,
    use_ema: bool = False,
) -> Any:
    """Train on real data + confident pseudo-labeled private examples."""
    combined = all_data + [{"text": x["text"], "entities": x["entities"]} for x in pseudo_data]
    print(
        f"\nAugmented training set: {len(combined)} examples "
        f"({len(all_data)} real + {len(pseudo_data)} pseudo)"
    )
    val_sample = random.sample(all_data, min(config.VAL_SAMPLE, len(all_data)))
    train_ds = NERDataset(combined, tokenizer)
    val_ds = NERDataset(val_sample, tokenizer)

    model = _new_model(model_name)
    trainer = _trainer(
        model, _build_training_args(output_dir, report_to="tensorboard"), train_ds, val_ds, use_ema
    )
    trainer.train()
    print("Augmented model training complete.")
    return model
