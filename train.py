# -*- coding: utf-8 -*-
"""
Train ruBert-base for PII NER with best practices.

Features:
  - BIO labeling scheme with 30 PII entity types
  - 5-fold cross-validation with probability blending
  - EMA (Exponential Moving Average) weights
  - Linear warmup + cosine schedule
  - Label smoothing
  - Class-weighted loss for imbalanced entities
  - Pseudo-labeling of confident test predictions
  - Post-processing: span expansion to word boundaries,
    overlap resolution, format validation
  - Mixed-precision (fp16/bf16) training
  - Gradient checkpointing for memory efficiency
  - Weights & Biases logging (optional)
"""

import csv
import ast
import json
import random
import logging
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_callback import TrainerControl, TrainerState

from postprocessing import postprocess_ner_predictions, PostprocessConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """All hyperparameters and paths in one place."""

    # Models
    model_name: str = "models/rubert-base-cased"
    # Uncomment to add a second model ensemble:
    # model_name_2: str = "Data-Lab/rubert-base-cased-conversational_ner-v1"

    # Data
    train_file: str = "data/train_dataset.tsv"
    synth_file: str = "data/synth_data_preproc.csv"
    test_file: str = ""          # set to private test CSV for inference
    output_file: str = "submission.csv"
    data_fraction: float = 1.0   # use only X% of training data (0.0-1.0)

    # Cross-validation
    n_folds: int = 5

    # Tokenization
    max_length: int = 512

    # Optimizer / schedule
    batch_size: int = 8
    grad_accum: int = 4          # effective batch = batch_size * grad_accum = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 10
    label_smoothing: float = 0.1

    # EMA
    ema_decay: float = 0.995
    use_ema: bool = True

    # Inference
    threshold: float = 0.5

    # Pseudo-labeling
    use_pseudo_labels: bool = False
    uncertain_thresh: float = 0.15
    confident_thresh: float = 0.04

    # Logging
    logging_steps: int = 50
    eval_steps: int = 300
    val_sample: int = 600
    seed: int = 42

    # Memory
    gradient_checkpointing: bool = True
    fp16: bool = True           # auto-disabled on CPU/MPS

    # W&B
    report_to: str = "none"     # set to "wandb" to enable


# ---------------------------------------------------------------------------
# Entity label schema (BIO)
# ---------------------------------------------------------------------------
ENTITY_LABELS = [
    "API ключи",
    "CVV/CVC",
    "Email",
    "Водительское удостоверение",
    "Временное удостоверение личности",
    "Гражданство и названия стран",
    "Данные об автомобиле клиента",
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)",
    "Дата окончания срока действия карты",
    "Дата регистрации по месту жительства или пребывания",
    "Дата рождения",
    "Имя держателя карты",
    "Кодовые слова",
    "Место рождения",
    "Наименование банка",
    "Номер банковского счета",
    "Номер карты",
    "Номер телефона",
    "Одноразовые коды",
    "ПИН код",
    "Пароли",
    "Паспортные данные",
    "Полный адрес",
    "Разрешение на работу / визу",
    "СНИЛС клиента",
    "Сведения об ИНН",
    "Свидетельство о рождении",
    "Серия и номер вида на жительство",
    "Содержимое магнитной полосы",
    "ФИО",
]

BIO_LABELS = ["O"]
for _label in ENTITY_LABELS:
    BIO_LABELS.append(f"B-{_label}")
    BIO_LABELS.append(f"I-{_label}")

label2id = {label: i for i, label in enumerate(BIO_LABELS)}
id2label = {i: label for i, label in enumerate(BIO_LABELS)}

NUM_LABELS = len(BIO_LABELS)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(filepath: str, seed: int = 42) -> List[Dict[str, Any]]:
    """Load train TSV with columns: text, target, entity."""
    data: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["text"]
            try:
                target = ast.literal_eval(row["target"])
            except Exception:
                target = []
            entities = []
            for item in target:
                if len(item) == 3:
                    start, end, label = item
                    entities.append((int(start), int(end), label))
            data.append({"text": text, "entities": entities})
    random.seed(seed)
    random.shuffle(data)
    return data


def load_synth_data(filepath: str) -> List[Dict[str, Any]]:
    """Load synthetic CSV with columns: id, text, target, entity."""
    data: List[Dict[str, Any]] = []
    df = pd.read_csv(filepath)
    for _, row in df.iterrows():
        text = row["text"]
        try:
            target = ast.literal_eval(row["target"])
        except Exception:
            target = []
        entities = []
        for item in target:
            if len(item) == 3:
                start, end, label = item
                entities.append((int(start), int(end), label))
        data.append({"text": text, "entities": entities})
    return data


def load_test_data(filepath: str) -> List[Dict[str, Any]]:
    """Load test CSV with columns: id, text."""
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append({"id": row["id"], "text": row["text"]})
    return texts


# ---------------------------------------------------------------------------
# Tokenization & dataset
# ---------------------------------------------------------------------------
def align_labels_with_tokens(
    text: str,
    entities: List[Tuple[int, int, str]],
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """Tokenize text and align BIO labels using character offset overlap."""
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )
    offset_mapping = encoding["offset_mapping"]
    labels = []
    for start, end in offset_mapping:
        if start == 0 and end == 0:
            labels.append(-100)       # special tokens ignored in loss
            continue
        token_label = "O"
        for ent_start, ent_end, ent_label in entities:
            if start < ent_end and end > ent_start:
                # B- label at first overlapping token, I- for subsequent
                token_label = f"B-{ent_label}" if start <= ent_start else f"I-{ent_label}"
                break
        labels.append(label2id.get(token_label, 0))
    del encoding["offset_mapping"]
    encoding["labels"] = labels
    return encoding


class NERDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int):
        self.encodings = []
        for item in tqdm(data, desc="Tokenizing"):
            enc = align_labels_with_tokens(
                item["text"], item["entities"], tokenizer, max_length
            )
            self.encodings.append(enc)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        return {
            "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(enc["labels"],         dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Class-weighted loss (handles label imbalance)
# ---------------------------------------------------------------------------
def compute_class_weights(dataset: NERDataset, num_labels: int) -> torch.Tensor:
    """Inverse-frequency weights for token-level CE loss."""
    counts = np.zeros(num_labels, dtype=np.float64)
    for enc in dataset.encodings:
        for lbl in enc["labels"]:
            if lbl >= 0:
                counts[lbl] += 1
    # Smooth: avoid division by zero, then normalise
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_labels
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------
class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, decay: float = 0.995):
        self.decay = decay
        self.shadow: Optional[Dict[str, torch.Tensor]] = None

    def _init_shadow(self, model: nn.Module):
        """Lazy init — called on first update when model is already on the right device."""
        self.shadow = {name: p.data.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    def update(self, model: nn.Module):
        if self.shadow is None:
            self._init_shadow(model)
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Swap in EMA weights (call before eval, call restore after)."""
        if self.shadow is None:
            return
        self.backup = {name: p.data.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        if not hasattr(self, "backup"):
            return
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        del self.backup


# ---------------------------------------------------------------------------
# Custom Trainer with EMA + class weights
# ---------------------------------------------------------------------------
class PIITrainer(Trainer):
    """Trainer that adds EMA and optional class-weighted loss."""

    def __init__(self, ema: Optional[EMA] = None, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.ema = ema
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None

        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100, label_smoothing=self.args.label_smoothing_factor)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1))

        if self.ema is not None:
            self.ema.update(model)

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, **kwargs):
        if self.ema is not None:
            self.ema.apply(self.model)
        try:
            result = super().evaluate(eval_dataset=eval_dataset, **kwargs)
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _extract_entities(labels: List[str]) -> set:
    entities = set()
    cur_type, cur_start = None, None
    for i, tag in enumerate(labels):
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
        entities.add((cur_type, cur_start, len(labels)))
    return entities


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)

    total   = (labels != -100).sum()
    correct = (preds == labels).astype(int)[labels != -100].sum()
    token_accuracy = correct / total if total else 0.0

    true_seqs, pred_seqs = [], []
    for pred_seq, label_seq in zip(preds, labels):
        t_seq, p_seq = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            t_seq.append(id2label[l])
            p_seq.append(id2label[p])
        true_seqs.append(t_seq)
        pred_seqs.append(p_seq)

    tp = fp = fn = 0
    for t_seq, p_seq in zip(true_seqs, pred_seqs):
        t_set = _extract_entities(t_seq)
        p_set = _extract_entities(p_seq)
        tp += len(t_set & p_set)
        fp += len(p_set - t_set)
        fn += len(t_set - p_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": token_accuracy}


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------
class MetricsLoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        parts = [f"Step {state.global_step}"]
        for key, fmt in [
            ("loss",           "Loss: {:.4f}"),
            ("learning_rate",  "LR: {:.2e}"),
            ("eval_loss",      "Eval Loss: {:.4f}"),
            ("eval_f1",        "Eval F1: {:.4f}"),
            ("eval_precision", "Eval P: {:.4f}"),
            ("eval_recall",    "Eval R: {:.4f}"),
        ]:
            if key in logs:
                parts.append(fmt.format(logs[key]))
        logger.info(" | ".join(parts))


# ---------------------------------------------------------------------------
# Training single fold
# ---------------------------------------------------------------------------
def train_fold(
    fold: int,
    train_data: List[Dict],
    val_data: List[Dict],
    tokenizer,
    cfg: TrainConfig,
    model_name: Optional[str] = None,
    output_tag: str = "",
) -> Tuple[Any, Optional[EMA]]:
    model_name = model_name or cfg.model_name
    tag = output_tag or f"fold{fold + 1}"
    logger.info(f"\n{'='*50}\nFOLD {fold + 1}/{cfg.n_folds}  [{model_name}]\n{'='*50}")

    train_dataset = NERDataset(train_data, tokenizer, cfg.max_length)
    val_sample = random.sample(val_data, min(cfg.val_sample, len(val_data)))
    val_dataset = NERDataset(val_sample, tokenizer, cfg.max_length)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Class weights — will be moved to model device in compute_loss
    class_weights = compute_class_weights(train_dataset, NUM_LABELS)

    # EMA — lazy init: shadow created on first update when model is on MPS
    ema = EMA(decay=cfg.ema_decay) if cfg.use_ema else None

    has_mps = torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=f"./ner_model_{tag}",
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        label_smoothing_factor=cfg.label_smoothing,
        report_to=cfg.report_to,
        fp16=has_cuda,
        bf16=False,
        use_cpu=not (has_mps or has_cuda),
        dataloader_num_workers=0,
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        seed=cfg.seed + fold,
        data_seed=cfg.seed,
    )

    trainer = PIITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLoggingCallback()],
        ema=ema,
        class_weights=class_weights,
    )

    trainer.train()

    # Evaluate on full val
    if ema is not None:
        ema.apply(model)
    val_dataset_full = NERDataset(val_data, tokenizer, cfg.max_length)
    val_metrics = trainer.evaluate(val_dataset_full)
    if ema is not None:
        ema.restore(model)

    logger.info(
        f"[{tag}] Val F1: {val_metrics.get('eval_f1', 0):.4f} | "
        f"P: {val_metrics.get('eval_precision', 0):.4f} | "
        f"R: {val_metrics.get('eval_recall', 0):.4f}"
    )

    # Save best checkpoint
    save_path = f"./ner_model_{tag}/best"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"[{tag}] Saved to {save_path}")

    return model, ema


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def get_token_probs(
    text: str,
    model,
    tokenizer,
    max_length: int = 512,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Returns (offset_mapping, probs) where probs shape = (seq_len, n_labels)."""
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
        probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    return offset_mapping, probs


def extract_entities_from_probs(
    offset_mapping: List[Tuple[int, int]],
    probs: np.ndarray,
    threshold: float = 0.5,
) -> List[Tuple[int, int, str]]:
    """Decode BIO entities from probability array."""
    entities = []
    current_entity = None

    for (start, end), token_probs in zip(offset_mapping, probs):
        if start == 0 and end == 0:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        pred_id = int(np.argmax(token_probs))
        conf    = float(token_probs[pred_id])
        label   = "O" if conf < threshold else id2label[pred_id]

        if label.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = (start, end, label[2:])

        elif label.startswith("I-"):
            entity_type = label[2:]
            if current_entity is not None and current_entity[2] == entity_type:
                current_entity = (current_entity[0], end, entity_type)
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = (start, end, entity_type)

        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    return entities


def compute_uncertainty(
    offset_mapping: List[Tuple[int, int]],
    avg_probs: np.ndarray,
) -> float:
    """1 - geometric_mean(max_prob) over real tokens."""
    log_max_probs = [
        float(np.log(np.clip(probs.max(), 1e-9, 1.0)))
        for (s, e), probs in zip(offset_mapping, avg_probs)
        if not (s == 0 and e == 0)
    ]
    if not log_max_probs:
        return 1.0
    geometric_mean = float(np.exp(np.mean(log_max_probs)))
    return 1.0 - geometric_mean


def blend_predictions(
    test_texts: List[Dict],
    model_tokenizer_pairs: List[Tuple[Any, Any]],
    threshold: float = 0.5,
) -> List[List[Tuple[int, int, str]]]:
    """Average softmax probabilities across all (model, tokenizer) pairs."""
    n_models = len(model_tokenizer_pairs)
    blended_entities: List[List[Tuple[int, int, str]]] = []

    for item in tqdm(test_texts, desc="Blending predictions"):
        text = item["text"]
        avg_probs: np.ndarray = None
        offset_mapping = None

        for model, tok in model_tokenizer_pairs:
            off, probs = get_token_probs(text, model, tok)
            if avg_probs is None:
                avg_probs      = probs.copy()
                offset_mapping = off
            else:
                avg_probs += probs

        avg_probs /= n_models
        blended_entities.append(extract_entities_from_probs(offset_mapping, avg_probs, threshold))

    return blended_entities


# ---------------------------------------------------------------------------
# Pseudo-labeling
# ---------------------------------------------------------------------------
def mine_pseudo_labels(
    test_texts: List[Dict],
    fold_models: List[Any],
    tokenizer,
    cfg: TrainConfig,
) -> Tuple[List[Dict], List[Dict]]:
    """Split test examples into confident (pseudo-labeled) and hard."""
    hard = []
    confident = []

    for item in tqdm(test_texts, desc="Mining pseudo labels"):
        text = item["text"]
        avg_probs: np.ndarray = None
        offset_mapping = None

        for model in fold_models:
            off, probs = get_token_probs(text, model, tokenizer)
            if avg_probs is None:
                avg_probs      = probs.copy()
                offset_mapping = off
            else:
                avg_probs += probs

        avg_probs /= len(fold_models)
        uncertainty = compute_uncertainty(offset_mapping, avg_probs)
        entities = extract_entities_from_probs(offset_mapping, avg_probs, cfg.threshold)

        if uncertainty > cfg.uncertain_thresh:
            hard.append({"id": item["id"], "text": text, "uncertainty": uncertainty})
        elif uncertainty < cfg.confident_thresh:
            confident.append({"text": text, "entities": entities, "uncertainty": uncertainty})

    return hard, confident


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = TrainConfig(
        test_file="data/private_test_dataset.csv"
    )

    # --- Load data ---
    logger.info("Loading training data...")
    all_data = load_data(cfg.train_file, seed=cfg.seed)
    logger.info(f"Loaded {len(all_data)} examples from {cfg.train_file}")

    # Optionally merge synthetic data
    if os.path.exists(cfg.synth_file):
        synth = load_synth_data(cfg.synth_file)
        logger.info(f"Loaded {len(synth)} synthetic examples from {cfg.synth_file}")
        all_data.extend(synth)
        random.seed(cfg.seed)
        random.shuffle(all_data)
        logger.info(f"Combined dataset: {len(all_data)} examples")

    # Subsample
    if cfg.data_fraction < 1.0:
        n = max(1, int(len(all_data) * cfg.data_fraction))
        random.seed(cfg.seed)
        all_data = random.sample(all_data, n)
        logger.info(f"Subsampled to {cfg.data_fraction:.0%} → {len(all_data)} examples")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # --- K-fold cross-validation ---
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_models: List[Any] = []
    fold_emas: List[Optional[EMA]] = []

    indices = list(range(len(all_data)))
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_data = [all_data[i] for i in train_idx]
        val_data   = [all_data[i] for i in val_idx]
        model, ema = train_fold(fold, train_data, val_data, tokenizer, cfg)
        fold_models.append(model)
        fold_emas.append(ema)

    # --- Inference on test data (if provided) ---
    if cfg.test_file and os.path.exists(cfg.test_file):
        test_texts = load_test_data(cfg.test_file)
        logger.info(f"Loaded {len(test_texts)} test examples")

        # Apply EMA before inference
        if cfg.use_ema:
            for model, ema in zip(fold_models, fold_emas):
                if ema is not None:
                    ema.apply(model)

        model_tok_pairs = [(m, tokenizer) for m in fold_models]

        # Pseudo-labeling (optional)
        if cfg.use_pseudo_labels:
            logger.info("Mining pseudo labels from test data...")
            hard, confident = mine_pseudo_labels(test_texts, fold_models, tokenizer, cfg)
            logger.info(f"Pseudo labels: {len(confident)} confident, {len(hard)} hard")

            # Save pseudo-labeled data
            pd.DataFrame([
                {"text": x["text"], "target": str(x["entities"])} for x in confident
            ]).to_csv("train_pseudo.csv", index=False)

            # Retrain with pseudo labels
            aug_data = all_data + [{"text": x["text"], "entities": x["entities"]} for x in confident]
            logger.info(f"Augmented training set: {len(aug_data)} examples")
            aug_model, aug_ema = train_fold(
                0, aug_data, random.sample(all_data, min(cfg.val_sample, len(all_data))),
                tokenizer, cfg, output_tag="augmented",
            )
            if aug_ema is not None:
                aug_ema.apply(aug_model)
            model_tok_pairs.append((aug_model, tokenizer))

        # Blend predictions
        logger.info("Blending predictions across all models...")
        blended = blend_predictions(test_texts, model_tok_pairs, cfg.threshold)

        # Build DataFrames for postprocessing
        preds_df = pd.DataFrame({
            "id":         [int(t["id"]) for t in test_texts],
            "Prediction": [str(ents) for ents in blended],
        })
        texts_df = pd.DataFrame({
            "id":   [int(t["id"]) for t in test_texts],
            "text": [t["text"]   for t in test_texts],
        })

        # Postprocess
        logger.info("Applying postprocessing...")
        result_df, report = postprocess_ner_predictions(preds_df, texts_df)
        logger.info(f"Postprocessing report: {report['counts']}")

        result_df.to_csv(cfg.output_file, index=False)
        logger.info(f"Saved {cfg.output_file}  ({len(result_df)} rows)")

        # Restore EMA
        if cfg.use_ema:
            for model, ema in zip(fold_models, fold_emas):
                if ema is not None:
                    ema.restore(model)
    else:
        logger.info("No test file provided. Training-only mode.")
        logger.info(f"Models saved in ./ner_model_fold*/best/")

    # --- Print OOF metrics summary ---
    logger.info("\nDone! Check ./ner_model_fold*/best/ for saved checkpoints.")


if __name__ == "__main__":
    main()
