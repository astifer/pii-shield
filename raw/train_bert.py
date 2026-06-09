# -*- coding: utf-8 -*-
"""
Train ai-forever/ruBert-base for PII NER with 5-fold cross-validation.
Features: EMA weights, linear warmup, probability blending across folds,
postprocessing via postprocessing.py.
"""

import csv
import ast
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
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
)
from transformers.trainer_callback import TrainerControl, TrainerState

from postprocessing import postprocess_ner_predictions, PostprocessConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME   = "DeepPavlov/rubert-base-cased"
MODEL_NAME_2 = "Data-Lab/rubert-base-cased-conversational_ner-v1"
TRAIN_FILE   = "data/train_dataset.tsv"
TEST_FILE    = "data/private_test_dataset.csv"
OUTPUT_FILE  = "submission.csv"

N_FOLDS      = 3
N_FOLDS_2    = 2
MAX_LENGTH   = 512
BATCH_SIZE   = 8   # per-device; effective batch = 32 via gradient accumulation
GRAD_ACCUM   = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS   = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
THRESHOLD    = 0.5
UNCERTAIN_THRESH = 0.15   # mean(1-max_prob) > this → hard/uncertain
CONFIDENT_THRESH = 0.04   # mean(1-max_prob) < this → confident pseudo-label
LOGGING_STEPS = 50
EVAL_STEPS    = 300
VAL_SAMPLE   = 600   # examples used for validation during training

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

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
id2label  = {i: label for i, label in enumerate(BIO_LABELS)}

print(f"Number of BIO labels: {len(BIO_LABELS)}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(filepath: str, seed: int = 42) -> List[Dict[str, Any]]:
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


def load_test_data(filepath: str) -> List[Dict[str, Any]]:
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append({"id": row["id"], "text": row["text"]})
    return texts

# ---------------------------------------------------------------------------
# Tokenisation & dataset
# ---------------------------------------------------------------------------
def align_labels_with_tokens(
    text: str,
    entities: List[Tuple[int, int, str]],
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
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
            labels.append(-100)
            continue
        token_label = "O"
        for ent_start, ent_end, ent_label in entities:
            if start < ent_end and end > ent_start:
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
# Metrics
# ---------------------------------------------------------------------------
def _extract_entities(labels: List[str]):
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
        print(" | ".join(parts))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_fold(
    fold: int,
    train_data: List[Dict],
    val_data: List[Dict],
    tokenizer,
    model_name: str = MODEL_NAME,
    output_tag: str = "",
) -> Any:
    tag = output_tag or f"fold{fold + 1}"
    n_folds_display = N_FOLDS if model_name == MODEL_NAME else N_FOLDS_2
    print(f"\n{'='*50}\nFOLD {fold + 1}/{n_folds_display}  [{model_name}]\n{'='*50}")

    train_dataset = NERDataset(train_data, tokenizer, MAX_LENGTH)

    # Small fixed val sample for fast evaluation during training
    val_sample = random.sample(val_data, min(VAL_SAMPLE, len(val_data)))
    val_dataset = NERDataset(val_sample, tokenizer, MAX_LENGTH)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(BIO_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=f"./ner_model_{tag}",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        label_smoothing_factor=0.1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLoggingCallback()],
    )

    trainer.train()

    # --- Train F1 on a small sample ---
    train_sample = random.sample(train_data, min(VAL_SAMPLE, len(train_data)))
    train_eval_dataset = NERDataset(train_sample, tokenizer, MAX_LENGTH)
    train_metrics = trainer.evaluate(train_eval_dataset)
    print(f"[{tag}] Train F1: {train_metrics.get('eval_f1', 0):.4f} | "
          f"Train P: {train_metrics.get('eval_precision', 0):.4f} | "
          f"Train R: {train_metrics.get('eval_recall', 0):.4f}")

    print(f"[{tag}] training complete.")
    return model

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def get_token_probs(
    text: str,
    model,
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Returns (offset_mapping, probs) where probs has shape (seq_len, n_labels)."""
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
) -> List[Tuple[int, int, str]]:
    """Decode BIO entities from averaged probability array."""
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
        label   = "O" if conf < THRESHOLD else id2label[pred_id]

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


def blend_predictions(
    test_texts: List[Dict],
    model_tokenizer_pairs: List[Tuple[Any, Any]],
) -> Tuple[
    List[List[Tuple[int, int, str]]],          # blended entities per example
    List[List[Dict]],                           # per_model[model_idx][example_idx]
]:
    """Average softmax probabilities across all (model, tokenizer) pairs in one pass.

    Each per-model entry is a dict with keys:
        entities      – decoded entity list
        uncertainty   – geometric-mean sequence uncertainty score
    """
    n_models = len(model_tokenizer_pairs)
    blended_entities: List[List[Tuple[int, int, str]]] = []
    per_model: List[List[Dict]] = [[] for _ in range(n_models)]

    for item in tqdm(test_texts, desc="Blending predictions"):
        text = item["text"]
        avg_probs: np.ndarray = None
        offset_mapping = None
        model_results: List[Tuple] = []

        for model, tok in model_tokenizer_pairs:
            off, probs = get_token_probs(text, model, tok)
            model_results.append((off, probs))
            if avg_probs is None:
                avg_probs      = probs.copy()
                offset_mapping = off
            else:
                avg_probs += probs

        avg_probs /= n_models
        blended_entities.append(extract_entities_from_probs(offset_mapping, avg_probs))

        for i, (off, probs) in enumerate(model_results):
            per_model[i].append({
                "entities":    extract_entities_from_probs(off, probs),
                "uncertainty": compute_uncertainty(off, probs),
            })

    return blended_entities, per_model


# ---------------------------------------------------------------------------
# Uncertainty mining helpers
# ---------------------------------------------------------------------------
def compute_uncertainty(
    offset_mapping: List[Tuple[int, int]],
    avg_probs: np.ndarray,
) -> float:
    """Returns 1 - geometric_mean(max_prob) over real (non-special) tokens.

    Geometric mean = exp(mean(log(max_probs))) — the standard sequence
    probability score (cumulative multiplication, length-normalised).
    """
    log_max_probs = [
        float(np.log(np.clip(probs.max(), 1e-9, 1.0)))
        for (s, e), probs in zip(offset_mapping, avg_probs)
        if not (s == 0 and e == 0)
    ]
    if not log_max_probs:
        return 1.0
    geometric_mean = float(np.exp(np.mean(log_max_probs)))
    return 1.0 - geometric_mean


def get_blended_probs(
    text: str,
    models: List[Any],
    tokenizer,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Blend probabilities from multiple models for a single text."""
    avg_probs: np.ndarray = None
    offset_mapping = None
    for model in models:
        off, probs = get_token_probs(text, model, tokenizer)
        if avg_probs is None:
            avg_probs = probs.copy()
            offset_mapping = off
        else:
            avg_probs += probs
    avg_probs /= len(models)
    return offset_mapping, avg_probs


def mine_train_examples(
    all_data: List[Dict],
    fold_models: List[Any],
    oof_val_indices: List[np.ndarray],
    tokenizer,
) -> List[Dict]:
    """For each train example, use its OOF model to compute honest uncertainty."""
    # Build map: example_index -> fold_index
    idx_to_fold = {}
    for fold_idx, val_idx in enumerate(oof_val_indices):
        for i in val_idx:
            idx_to_fold[int(i)] = fold_idx

    hard = []
    for ex_idx, item in enumerate(tqdm(all_data, desc="Mining train examples")):
        fold_idx = idx_to_fold.get(ex_idx)
        if fold_idx is None:
            continue  # shouldn't happen with KFold

        model = fold_models[fold_idx]
        offset_mapping, probs = get_token_probs(item["text"], model, tokenizer)

        uncertainty = compute_uncertainty(offset_mapping, probs)
        predicted_entities = set(extract_entities_from_probs(offset_mapping, probs))
        true_entities = set(
            (s, e, l) for s, e, l in item["entities"]
        )
        is_incorrect = predicted_entities != true_entities

        if uncertainty > UNCERTAIN_THRESH or is_incorrect:
            hard.append({
                "text": item["text"],
                "target": str(item["entities"]),
                "predicted": str(list(predicted_entities)),
                "uncertainty_score": uncertainty,
                "is_incorrect": is_incorrect,
            })

    return hard


def mine_private_examples(
    test_texts: List[Dict],
    fold_models: List[Any],
    tokenizer,
) -> Tuple[List[Dict], List[Dict]]:
    """Blend all fold models for each private example; split into hard and confident."""
    hard = []
    confident = []

    for item in tqdm(test_texts, desc="Mining private examples"):
        text = item["text"]
        offset_mapping, avg_probs = get_blended_probs(text, fold_models, tokenizer)
        uncertainty = compute_uncertainty(offset_mapping, avg_probs)
        entities = extract_entities_from_probs(offset_mapping, avg_probs)

        if uncertainty > UNCERTAIN_THRESH:
            hard.append({
                "id": item["id"],
                "text": text,
                "predicted": str(entities),
                "uncertainty_score": uncertainty,
            })
        elif uncertainty < CONFIDENT_THRESH:
            confident.append({
                "id": item["id"],
                "text": text,
                "entities": entities,
                "uncertainty_score": uncertainty,
            })

    return hard, confident


def train_augmented_model(
    all_data: List[Dict],
    pseudo_data: List[Dict],
    tokenizer,
) -> Any:
    """Train a model on all_data + confident pseudo-labeled private examples."""
    combined = all_data + [
        {"text": x["text"], "entities": x["entities"]} for x in pseudo_data
    ]
    print(f"\nAugmented training set: {len(combined)} examples "
          f"({len(all_data)} real + {len(pseudo_data)} pseudo)")

    val_sample = random.sample(all_data, min(VAL_SAMPLE, len(all_data)))
    train_data = combined

    train_dataset = NERDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset   = NERDataset(val_sample,  tokenizer, MAX_LENGTH)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(BIO_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="./ner_model_augmented",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        label_smoothing_factor=0.1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLoggingCallback()],
    )

    trainer.train()
    print("Augmented model training complete.")
    return model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading training data...")
    all_data = load_data(TRAIN_FILE)
    print(f"Loaded {len(all_data)} examples")

    print("Loading test data...")
    test_texts = load_test_data(TEST_FILE)
    print(f"Loaded {len(test_texts)} test examples")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # --- K-fold cross-validation ---
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_models: List[Any] = []
    oof_val_indices: List[np.ndarray] = []

    indices = list(range(len(all_data)))
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_data = [all_data[i] for i in train_idx]
        val_data   = [all_data[i] for i in val_idx]

        model = train_fold(fold, train_data, val_data, tokenizer)
        fold_models.append(model)
        oof_val_indices.append(val_idx)

    # --- Mine hard/uncertain train examples (OOF) ---
    print("\nMining hard train examples (OOF uncertainty)...")
    hard_train = mine_train_examples(all_data, fold_models, oof_val_indices, tokenizer)
    pd.DataFrame(hard_train).to_csv("hard_examples_train.csv", index=False)
    print(f"Saved hard_examples_train.csv ({len(hard_train)} examples)")

    # --- Mine private examples ---
    print("\nMining private examples (blended uncertainty)...")
    hard_private, confident_private = mine_private_examples(test_texts, fold_models, tokenizer)
    pd.DataFrame(hard_private).to_csv("hard_examples_private.csv", index=False)
    print(f"Saved hard_examples_private.csv ({len(hard_private)} hard, {len(confident_private)} confident)")

    # --- Save pseudo-labeled confident private examples ---
    train_private_rows = [{"text": x["text"], "target": str(x["entities"])} for x in confident_private]
    pd.DataFrame(train_private_rows).to_csv("train_private.csv", index=False)
    print(f"Saved train_private.csv ({len(train_private_rows)} pseudo-labeled examples)")

    # --- Train augmented model ---
    aug_model = train_augmented_model(all_data, confident_private, tokenizer)

    # --- Train 2-fold conversational model ---
    print(f"\nLoading tokenizer for {MODEL_NAME_2}...")
    tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME_2)
    print(f"Tokenizer2 vocab size: {tokenizer2.vocab_size}")

    kf2 = KFold(n_splits=N_FOLDS_2, shuffle=True, random_state=42)
    conv_models: List[Any] = []
    for fold, (train_idx, val_idx) in enumerate(kf2.split(indices)):
        train_data = [all_data[i] for i in train_idx]
        val_data   = [all_data[i] for i in val_idx]
        model = train_fold(
            fold, train_data, val_data, tokenizer2,
            model_name=MODEL_NAME_2,
            output_tag=f"conv_fold{fold + 1}",
        )
        conv_models.append(model)

    # --- Blend predictions: 3 rubert folds + augmented + 2 conv folds (single pass) ---
    model_tokenizer_pairs = (
        [(m, tokenizer)  for m in fold_models]
        + [(aug_model, tokenizer)]
        + [(m, tokenizer2) for m in conv_models]
    )
    model_names = (
        [f"fold{i+1}" for i in range(len(fold_models))]
        + ["augmented"]
        + [f"conv_fold{i+1}" for i in range(len(conv_models))]
    )

    print("\nBlending predictions across all models...")
    blended, per_model = blend_predictions(test_texts, model_tokenizer_pairs)

    # --- Save per-model predictions + uncertainty before blending ---
    blended_private_df = pd.DataFrame({"id": [int(t["id"]) for t in test_texts]})
    for name, model_data in zip(model_names, per_model):
        blended_private_df[f"pred_{name}"]        = [str(d["entities"])    for d in model_data]
        blended_private_df[f"uncertainty_{name}"] = [d["uncertainty"]      for d in model_data]
    blended_private_df.to_csv("blended_private.csv", index=False)
    print(f"Saved blended_private.csv ({len(blended_private_df)} rows, {len(model_names)} models)")

    # --- Build DataFrames for postprocessing ---
    preds_df = pd.DataFrame({
        "id":         [int(t["id"]) for t in test_texts],
        "Prediction": [str(ents) for ents in blended],
    })
    texts_df = pd.DataFrame({
        "id":   [int(t["id"]) for t in test_texts],
        "text": [t["text"]   for t in test_texts],
    })

    # --- Postprocess ---
    print("Applying postprocessing...")
    result_df, report = postprocess_ner_predictions(
        preds_df, texts_df, cfg=PostprocessConfig()
    )
    print("Postprocessing report (counts):", report["counts"])

    # --- Save submission ---
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {OUTPUT_FILE}  ({len(result_df)} rows)")

    # Quick format check
    sample = result_df.head(5)
    print("\nSample output:")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
