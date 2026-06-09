from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


@dataclass(frozen=True)
class PostprocessConfig:
    # --- Span expansion ---
    expand_to_word_boundaries: bool = True

    # Define "word boundary" stop chars. Keep '-' and '—' OUT if you want to allow hyphenated words.
    # In the original script they were included; that can over-trim things like "Санкт-Петербург".
    stop_chars: frozenset = frozenset(set(" \t\n.,;:!?()[]{}\"'«»"))

    # If True, also stops on hyphens/dashes; usually better False for Russian names/addresses.
    stop_on_dashes: bool = False

    # --- Basic removals ---
    remove_punct_only: bool = True
    # What counts as "punct only" after stripping whitespace:
    punct_only_re: re.Pattern = re.compile(r"^[\W_]+$", re.UNICODE)

    # --- Hypothesis 1 (targeted short-letter removals) ---
    remove_short_letter_spans: bool = True
    short_letter_max_len: int = 3
    labels_to_filter_short_letters: frozenset = frozenset({
        "Дата регистрации по месту жительства или пребывания",
        "Дата окончания срока действия карты",
    })

    # --- Overlap resolution ---
    remove_overlaps: bool = True
    # Prefer longer spans if overlapping; ties broken by earlier start, then label sort.
    prefer_longer_on_overlap: bool = True

    # --- Optional format validation filtering (stronger than “suspicious”, but can boost precision) ---
    filter_invalid_by_format: bool = False


def postprocess_ner_predictions(
    preds_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    *,
    pred_id_col: str = "id",
    pred_col: str = "Prediction",
    text_id_col: str = "id",
    text_col: str = "text",
    cfg: PostprocessConfig = PostprocessConfig(),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Post-process NER spans stored as strings like:
        "[(start, end, label), ...]"  or  "[(start, end, 'Label'), ...]"
    Returns:
        (result_df, report)
    where result_df has columns [pred_id_col, pred_col] and report contains
    counts + samples of changes/removals.

    Notes:
    - Uses a safer parser (ast.literal_eval) and normalizes to List[Tuple[int,int,str]].
    - Expansion is to nearest boundaries based on cfg.stop_chars (and optional dashes).
    - Removes punctuation/whitespace-only spans aggressively (cfg.remove_punct_only).
    - Removes known short-letter false positives for selected labels (cfg.remove_short_letter_spans).
    - Resolves overlaps deterministically (cfg.remove_overlaps).
    - Optional: drops entities failing label-specific format checks (cfg.filter_invalid_by_format).
    """

    # ---------- helpers ----------
    dash_chars = set("-—–") if cfg.stop_on_dashes else set()
    stop_chars = set(cfg.stop_chars) | dash_chars

    def _safe_parse_entities(obj: Any) -> List[Tuple[int, int, str]]:
        if obj is None:
            return []
        if isinstance(obj, (list, tuple)):
            raw = obj
        else:
            s = str(obj).strip()
            if not s or s == "[]":
                return []
            try:
                raw = ast.literal_eval(s)
            except Exception:
                return []

        ents: List[Tuple[int, int, str]] = []
        if not isinstance(raw, (list, tuple)):
            return []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            try:
                start = int(item[0])
                end = int(item[1])
                label = str(item[2])
            except Exception:
                continue
            if start < 0 or end < 0 or end <= start:
                continue
            ents.append((start, end, label))
        return ents

    def _expand_span(text: str, start: int, end: int) -> Tuple[int, int]:
        # Clamp
        n = len(text)
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if end <= start:
            return start, end

        new_start = start
        while new_start > 0 and text[new_start - 1] not in stop_chars:
            new_start -= 1

        new_end = end
        while new_end < n and text[new_end] not in stop_chars:
            new_end += 1

        return new_start, new_end

    def _is_punct_only(span_text: str) -> bool:
        t = span_text.strip()
        if not t:
            return True
        # Stronger than checking against a tiny list: catches ",", ")", "—", etc.
        return bool(cfg.punct_only_re.match(t))

    def _is_only_letters(span_text: str) -> bool:
        t = span_text.strip()
        return bool(t) and t.isalpha()

    def _should_remove_short_letter(span_text: str, label: str) -> bool:
        if not cfg.remove_short_letter_spans:
            return False
        if label not in cfg.labels_to_filter_short_letters:
            return False
        t = span_text.strip()
        if len(t) > cfg.short_letter_max_len:
            return False
        return _is_only_letters(t)

    def _validate_entity(text: str, label: str) -> Tuple[bool, str]:
        # Mostly taken from your notebook logic, lightly tightened.
        t = text.strip()

        if label == "Email":
            if "@" not in t:
                return False, "Email missing @"
            if not re.search(r"@[\w.-]+\.\w+", t):
                return False, "Invalid email format"

        elif label == "Номер телефона":
            digits = re.sub(r"\D", "", t)
            if len(digits) < 7:
                return False, f"Too few digits: {len(digits)}"

        elif label == "CVV/CVC":
            digits = re.sub(r"\D", "", t)
            if not (3 <= len(digits) <= 4):
                return False, f"CVV must be 3-4 digits, got {len(digits)}"

        elif label == "ПИН код":
            digits = re.sub(r"\D", "", t)
            if len(digits) != 4:
                return False, f"PIN must be 4 digits, got {len(digits)}"

        elif label == "ФИО":
            if not re.search(r"[а-яА-ЯёЁa-zA-Z]", t):
                return False, "FIO has no letters"
            if re.match(r"^[\d\s\-]+$", t):
                return False, "FIO is only digits/spaces"

        elif label == "СНИЛС клиента":
            digits = re.sub(r"\D", "", t)
            if len(digits) != 11:
                return False, f"SNILS must be 11 digits, got {len(digits)}"

        elif label == "Сведения об ИНН":
            digits = re.sub(r"\D", "", t)
            if len(digits) not in (10, 12):
                return False, f"INN must be 10 or 12 digits, got {len(digits)}"

        elif label == "Номер карты":
            digits = re.sub(r"\D", "", t)
            if not (13 <= len(digits) <= 19):
                return False, f"Card must be 13-19 digits, got {len(digits)}"

        elif label == "Номер банковского счета":
            digits = re.sub(r"\D", "", t)
            if len(digits) != 20:
                return False, f"Account must be 20 digits, got {len(digits)}"

        elif label == "Одноразовые коды":
            digits = re.sub(r"\D", "", t)
            if not (4 <= len(digits) <= 8):
                return False, f"OTP typically 4-8 digits, got {len(digits)}"

        elif label in ("Дата рождения", "Дата окончания срока действия карты"):
            if not re.search(r"\d", t):
                return False, "Date has no digits"

        return True, "OK"

    def _resolve_overlaps(
        entities: List[Tuple[int, int, str]]
    ) -> Tuple[List[Tuple[int, int, str]], List[Dict[str, Any]]]:
        """
        Deterministic overlap resolver:
        - Sort by start asc, length desc (so longer first at same start).
        - Maintain accepted spans; if overlap occurs, keep longer (or existing on tie).
        """
        removed_log: List[Dict[str, Any]] = []
        if not entities:
            return [], removed_log

        ents = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0]), str(x[2])))

        kept: List[Tuple[int, int, str]] = []
        for cur in ents:
            s, e, lab = cur
            overlapped_idx = None
            for i, ex in enumerate(kept):
                es, ee, elab = ex
                if s < ee and e > es:
                    overlapped_idx = i
                    break
            if overlapped_idx is None:
                kept.append(cur)
                continue

            ex = kept[overlapped_idx]
            es, ee, elab = ex
            cur_len = e - s
            ex_len = ee - es

            keep_current = False
            if cfg.prefer_longer_on_overlap:
                if cur_len > ex_len:
                    keep_current = True
            # tie-break: keep existing

            if keep_current:
                kept[overlapped_idx] = cur
                removed_log.append({
                    "removed": ex,
                    "kept": cur,
                    "reason": "overlap_keep_longer",
                })
            else:
                removed_log.append({
                    "removed": cur,
                    "kept": ex,
                    "reason": "overlap_drop_shorter_or_tie",
                })

        # Re-sort kept by start for nicer output consistency
        kept = sorted(kept, key=lambda x: (x[0], x[1], str(x[2])))
        return kept, removed_log

    # ---------- prepare id->text map for speed ----------
    # This avoids repeated .loc lookups inside loops.
    text_map = dict(zip(texts_df[text_id_col].tolist(), texts_df[text_col].tolist()))

    # ---------- main loop ----------
    out_rows: List[Dict[str, Any]] = []

    changes: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    removed_overlaps: List[Dict[str, Any]] = []
    removed_invalid: List[Dict[str, Any]] = []

    for _, row in preds_df.iterrows():
        pid = row[pred_id_col]
        pred_obj = row[pred_col]
        text = text_map.get(pid, "")
        entities = _safe_parse_entities(pred_obj)

        if not entities or not text:
            out_rows.append({pred_id_col: pid, pred_col: "[]"})
            continue

        # 1) basic filtering + optional expansion
        cleaned: List[Tuple[int, int, str]] = []
        for (s, e, lab) in entities:
            if s >= len(text):
                continue
            e = min(e, len(text))
            if e <= s:
                continue

            span_text = text[s:e]

            if cfg.remove_punct_only and _is_punct_only(span_text):
                removed.append({"id": pid, "label": lab, "span": (s, e), "text": span_text, "reason": "punct_only"})
                continue

            if _should_remove_short_letter(span_text, lab):
                removed.append({"id": pid, "label": lab, "span": (s, e), "text": span_text, "reason": "short_letter"})
                continue

            if cfg.expand_to_word_boundaries:
                ns, ne = _expand_span(text, s, e)
                if (ns, ne) != (s, e):
                    changes.append({
                        "id": pid,
                        "label": lab,
                        "old_span": (s, e),
                        "new_span": (ns, ne),
                        "old_text": span_text,
                        "new_text": text[ns:ne],
                        "reason": "expand_to_boundary",
                    })
                s, e = ns, ne

            cleaned.append((s, e, lab))

        # 2) resolve overlaps
        if cfg.remove_overlaps and cleaned:
            cleaned, ov_removed = _resolve_overlaps(cleaned)
            # Add texts for overlap logs
            for item in ov_removed:
                rs, re_, rlab = item["removed"]
                ks, ke_, klab = item["kept"]
                removed_overlaps.append({
                    "id": pid,
                    "removed": item["removed"],
                    "removed_text": text[rs:re_],
                    "kept": item["kept"],
                    "kept_text": text[ks:ke_],
                    "reason": item["reason"],
                })

        # 3) optional format validation filter
        if cfg.filter_invalid_by_format and cleaned:
            final: List[Tuple[int, int, str]] = []
            for (s, e, lab) in cleaned:
                span_text = text[s:e]
                ok, reason = _validate_entity(span_text, lab)
                if not ok:
                    removed_invalid.append({
                        "id": pid,
                        "label": lab,
                        "span": (s, e),
                        "text": span_text,
                        "reason": reason,
                    })
                else:
                    final.append((s, e, lab))
            cleaned = final

        out_rows.append({pred_id_col: pid, pred_col: str(cleaned)})

    result_df = pd.DataFrame(out_rows)

    report: Dict[str, Any] = {
        "config": cfg,
        "counts": {
            "rows_in": int(len(preds_df)),
            "rows_out": int(len(result_df)),
            "span_changes_expand": int(len(changes)),
            "removed_basic": int(len(removed)),
            "removed_overlaps": int(len(removed_overlaps)),
            "removed_invalid_format": int(len(removed_invalid)),
        },
        # keep logs, but you can drop these if you want it lighter
        "samples": {
            "changes_expand": changes[:25],
            "removed_basic": removed[:25],
            "removed_overlaps": removed_overlaps[:25],
            "removed_invalid_format": removed_invalid[:25],
        },
    }
    return result_df, report
