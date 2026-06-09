// Pure, DOM-free entity logic shared by the browser app and the Node tests.
// LABEL_TAGS (entity label -> placeholder) is generated from pii_ner.labels by
// pii-export-onnx, so it stays in sync with the trained label set.

import { LABEL_TAGS } from "./labels.generated.js";

export { LABEL_TAGS };

export function tagFor(label) {
  return LABEL_TAGS[label] || `[${label}]`;
}

// Merge tokens into entity spans, stitching adjacent same-label spans (one-char gaps
// between subwords). entity_group is set by aggregation_strategy "simple"; entity is
// the fallback for unaggregated output.
export function mergeSpans(tokens) {
  const spans = [];
  for (const t of tokens || []) {
    const label = t.entity_group || t.entity;
    if (!label || label === "O") continue;
    if (t.start == null || t.end == null) continue;
    const last = spans[spans.length - 1];
    if (last && last.label === label && t.start <= last.end + 1) {
      last.end = Math.max(last.end, t.end);
    } else {
      spans.push({ label, start: t.start, end: t.end });
    }
  }
  return spans;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// Original text with each span wrapped in <span class="entity">…</span>.
// Walks spans in order over the ORIGINAL text so offsets stay valid.
export function buildHighlighted(text, spans) {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  let out = "";
  let cursor = 0;
  for (const s of sorted) {
    out += escapeHtml(text.slice(cursor, s.start));
    out += `<span class="entity" title="${escapeHtml(s.label)}">${escapeHtml(
      text.slice(s.start, s.end)
    )}</span>`;
    cursor = s.end;
  }
  out += escapeHtml(text.slice(cursor));
  return out;
}

// Original text with each span replaced by its placeholder tag (e.g. [ФИО]).
export function buildObfuscated(text, spans) {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  let out = "";
  let cursor = 0;
  for (const s of sorted) {
    out += text.slice(cursor, s.start);
    out += tagFor(s.label);
    cursor = s.end;
  }
  out += text.slice(cursor);
  return out;
}
