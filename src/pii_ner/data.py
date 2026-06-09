"""Dataset loading and tokenisation for token-classification NER."""

from __future__ import annotations

import ast
import csv
import random
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import MAX_LENGTH, SEED
from .labels import label2id

Entity = tuple[int, int, str]
Example = dict[str, Any]


def load_train(filepath: str, seed: int = SEED) -> list[Example]:
    """Load the train TSV into ``[{"text", "entities"}, ...]``.

    Shuffled with a fixed seed so the KFold split is reproducible across train,
    mining and evaluation.
    """
    data: list[Example] = []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                target = ast.literal_eval(row["target"])
            except Exception:
                target = []
            entities: list[Entity] = []
            for item in target:
                if len(item) == 3:
                    start, end, label = item
                    entities.append((int(start), int(end), label))
            data.append({"text": row["text"], "entities": entities})
    random.seed(seed)
    random.shuffle(data)
    return data


def load_test(filepath: str) -> list[Example]:
    """Load the private test CSV into ``[{"id", "text"}, ...]``."""
    with open(filepath, encoding="utf-8") as f:
        return [{"id": row["id"], "text": row["text"]} for row in csv.DictReader(f)]


def align_labels_with_tokens(
    text: str,
    entities: list[Entity],
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Tokenise ``text`` and project char-span entities onto BIO token labels.

    Special tokens get ``-100`` (ignored by the loss); a token is ``B-`` when it
    starts at/after the entity start, else ``I-``.
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )
    labels: list[int] = []
    for start, end in encoding["offset_mapping"]:
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
    def __init__(self, data: list[Example], tokenizer, max_length: int = MAX_LENGTH):
        self.encodings = [
            align_labels_with_tokens(item["text"], item["entities"], tokenizer, max_length)
            for item in tqdm(data, desc="Tokenizing")
        ]

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.encodings[idx]
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(enc["labels"], dtype=torch.long),
        }
