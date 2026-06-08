# -*- coding: utf-8 -*-
"""
Inference module for PII Shield.

Loads the trained ModernBERT NER model once and exposes `detect(text, threshold)`
which returns the detected PII entities (with character spans) plus a masked
version of the text. The decoding logic mirrors `train.py`
(`get_token_probs` / `extract_entities_from_probs`) and the span clean-up mirrors
`postprocessing.py` (word-boundary expansion + overlap resolution).
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

MODEL_DIR = "ner_model_final"
MAX_LENGTH = 512

# Characters that mark a word boundary when expanding / trimming a span.
# Mirrors PostprocessConfig.stop_chars (dashes kept inside words).
_STOP_CHARS = frozenset(set(" \t\n.,;:!?()[]{}\"'«»"))

# Short, human-friendly mask tokens per entity type (label without B-/I- prefix).
ENTITY_SHORT: Dict[str, str] = {
    "API ключи": "API-КЛЮЧ",
    "CVV/CVC": "CVV",
    "Email": "EMAIL",
    "Водительское удостоверение": "ВОДИТ_УДОСТ",
    "Временное удостоверение личности": "ВРЕМ_УДОСТ",
    "Гражданство и названия стран": "ГРАЖДАНСТВО",
    "Данные об автомобиле клиента": "АВТО",
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)": "ОРГАНИЗАЦИЯ",
    "Дата окончания срока действия карты": "СРОК_КАРТЫ",
    "Дата регистрации по месту жительства или пребывания": "ДАТА_РЕГ",
    "Дата рождения": "ДАТА_РОЖД",
    "Имя держателя карты": "ДЕРЖАТЕЛЬ_КАРТЫ",
    "Кодовые слова": "КОДОВОЕ_СЛОВО",
    "Место рождения": "МЕСТО_РОЖД",
    "Наименование банка": "БАНК",
    "Номер банковского счета": "СЧЁТ",
    "Номер карты": "КАРТА",
    "Номер телефона": "ТЕЛЕФОН",
    "Одноразовые коды": "OTP",
    "ПИН код": "PIN",
    "Пароли": "ПАРОЛЬ",
    "Паспортные данные": "ПАСПОРТ",
    "Полный адрес": "АДРЕС",
    "Разрешение на работу / визу": "ВИЗА",
    "СНИЛС клиента": "СНИЛС",
    "Сведения об ИНН": "ИНН",
    "Свидетельство о рождении": "СВИД_РОЖД",
    "Серия и номер вида на жительство": "ВНЖ",
    "Содержимое магнитной полосы": "МАГ_ПОЛОСА",
    "ФИО": "ФИО",
}


def mask_token(label: str) -> str:
    return "[" + ENTITY_SHORT.get(label, label) + "]"


# --------------------------------------------------------------------------- #
#  Lazy model singleton
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_tokenizer = None
_model = None
_id2label: Dict[int, str] = {}


def load_model() -> None:
    """Load tokenizer + model into module globals (idempotent, thread-safe)."""
    global _tokenizer, _model, _id2label
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        model.eval()
        torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
        _tokenizer = tok
        _model = model
        # config.id2label keys may be str or int depending on load path
        _id2label = {int(k): v for k, v in model.config.id2label.items()}


def entity_types() -> List[str]:
    """Sorted list of the 30 base entity types the model can detect."""
    load_model()
    types = {lab[2:] for lab in _id2label.values() if lab != "O"}
    return sorted(types)


# --------------------------------------------------------------------------- #
#  Core inference
# --------------------------------------------------------------------------- #
def _get_token_probs(text: str) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    enc = _tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(_model.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = _model(**enc).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    return offsets, probs


def _decode_bio(
    offsets: List[Tuple[int, int]],
    probs: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, str, float]]:
    """BIO decode -> list of (start, end, label, mean_confidence)."""
    entities: List[Tuple[int, int, str, float]] = []
    cur_start = cur_end = None
    cur_label = None
    cur_confs: List[float] = []

    def flush():
        if cur_label is not None:
            entities.append(
                (cur_start, cur_end, cur_label, float(np.mean(cur_confs)))
            )

    for (start, end), token_probs in zip(offsets, probs):
        if start == 0 and end == 0:  # special token
            flush()
            cur_start = cur_end = cur_label = None
            cur_confs = []
            continue

        pred_id = int(np.argmax(token_probs))
        conf = float(token_probs[pred_id])
        label = "O" if conf < threshold else _id2label[pred_id]

        if label.startswith("B-"):
            flush()
            cur_start, cur_end, cur_label = start, end, label[2:]
            cur_confs = [conf]
        elif label.startswith("I-"):
            etype = label[2:]
            if cur_label == etype:
                cur_end = end
                cur_confs.append(conf)
            else:
                flush()
                cur_start, cur_end, cur_label = start, end, etype
                cur_confs = [conf]
        else:  # "O"
            flush()
            cur_start = cur_end = cur_label = None
            cur_confs = []

    flush()
    return entities


# --------------------------------------------------------------------------- #
#  Span clean-up (mirrors postprocessing.py defaults)
# --------------------------------------------------------------------------- #
def _trim(text: str, start: int, end: int) -> Tuple[int, int]:
    """Strip leading/trailing stop-chars from a span."""
    while start < end and text[start] in _STOP_CHARS:
        start += 1
    while end > start and text[end - 1] in _STOP_CHARS:
        end -= 1
    return start, end


def _expand_to_word_boundaries(text: str, start: int, end: int) -> Tuple[int, int]:
    n = len(text)
    while start > 0 and text[start - 1] not in _STOP_CHARS:
        start -= 1
    while end < n and text[end] not in _STOP_CHARS:
        end += 1
    return start, end


def _resolve_overlaps(
    ents: List[Tuple[int, int, str, float]]
) -> List[Tuple[int, int, str, float]]:
    """Keep longer span on overlap; ties keep the earlier-accepted one."""
    ordered = sorted(ents, key=lambda x: (x[0], -(x[1] - x[0]), x[2]))
    kept: List[Tuple[int, int, str, float]] = []
    for cur in ordered:
        s, e = cur[0], cur[1]
        clash = None
        for i, ex in enumerate(kept):
            if s < ex[1] and e > ex[0]:
                clash = i
                break
        if clash is None:
            kept.append(cur)
        elif (e - s) > (kept[clash][1] - kept[clash][0]):
            kept[clash] = cur
    return sorted(kept, key=lambda x: (x[0], x[1]))


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def detect(text: str, threshold: float = 0.5) -> Dict:
    """
    Run NER on `text` and return:
        {
          "text": <original text>,
          "entities": [{start, end, label, mask, text, confidence}, ...],
          "masked": <text with PII replaced by [TOKEN]>,
          "counts": {label: n, ...},
        }
    """
    load_model()
    text = text or ""
    if not text.strip():
        return {"text": text, "entities": [], "masked": text, "counts": {}}

    offsets, probs = _get_token_probs(text)
    raw = _decode_bio(offsets, probs, threshold)

    cleaned: List[Tuple[int, int, str, float]] = []
    for start, end, label, conf in raw:
        start, end = _expand_to_word_boundaries(text, start, end)
        start, end = _trim(text, start, end)
        if end <= start:
            continue
        cleaned.append((start, end, label, conf))

    cleaned = _resolve_overlaps(cleaned)

    entities = []
    counts: Dict[str, int] = {}
    for start, end, label, conf in cleaned:
        entities.append({
            "start": start,
            "end": end,
            "label": label,
            "mask": mask_token(label),
            "text": text[start:end],
            "confidence": round(conf, 4),
        })
        counts[label] = counts.get(label, 0) + 1

    # Build masked text (entities are non-overlapping & sorted by start).
    pieces = []
    cursor = 0
    for ent in entities:
        pieces.append(text[cursor:ent["start"]])
        pieces.append(ent["mask"])
        cursor = ent["end"]
    pieces.append(text[cursor:])
    masked = "".join(pieces)

    return {
        "text": text,
        "entities": entities,
        "masked": masked,
        "counts": counts,
    }
