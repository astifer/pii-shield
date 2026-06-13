// Unit tests for the pure entity logic. Run with:  node --test web/obfuscate.test.js
// Requires web/labels.generated.js (produced by `pii-export-onnx`).
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildHighlighted,
  buildObfuscated,
  escapeHtml,
  mergeSpans,
  tagFor,
} from "./obfuscate.js";

// Helper: build a transformers.js-style aggregated token result.
const tok = (entity_group, start, end) => ({ entity_group, start, end });

test("LABEL_TAGS includes the real PII entity types", () => {
  assert.equal(tagFor("ФИО"), "[ФИО]");
  assert.equal(tagFor("Полный адрес"), "[ПОЛНЫЙ_АДРЕС]");
  assert.equal(tagFor("Номер телефона"), "[НОМЕР_ТЕЛЕФОНА]");
});

test("tagFor falls back to [LABEL] for an unmapped label", () => {
  assert.equal(tagFor("Неизвестный тип"), "[Неизвестный тип]");
});

test("mergeSpans keeps every non-O entity type", () => {
  const spans = mergeSpans([
    tok("ФИО", 0, 4),
    tok("O", 5, 9),
    tok("Номер телефона", 10, 21),
  ]);
  assert.deepEqual(spans, [
    { label: "ФИО", start: 0, end: 4 },
    { label: "Номер телефона", start: 10, end: 21 },
  ]);
});

test("mergeSpans merges adjacent same-label tokens (touching or 1-char gap)", () => {
  const spans = mergeSpans([
    tok("ФИО", 0, 4),
    tok("ФИО", 5, 12),
    tok("ФИО", 12, 18),
  ]);
  assert.deepEqual(spans, [{ label: "ФИО", start: 0, end: 18 }]);
});

test("mergeSpans does NOT merge across a real gap or different labels", () => {
  const spans = mergeSpans([
    tok("ФИО", 0, 4),
    tok("ФИО", 10, 14),
    tok("Полный адрес", 14, 20),
  ]);
  assert.equal(spans.length, 3);
});

test("mergeSpans tolerates the 'entity' key and missing offsets / empty input", () => {
  assert.deepEqual(
    mergeSpans([
      { entity: "ФИО", start: 0, end: 4 },
      { entity: "ФИО", start: null, end: 4 }, // null offsets -> skipped
    ]),
    [{ label: "ФИО", start: 0, end: 4 }]
  );
  assert.deepEqual(mergeSpans([]), []);
  assert.deepEqual(mergeSpans(undefined), []);
});

test("buildObfuscated replaces each span with its type tag", () => {
  const text = "Иван живёт в Москве";
  const spans = [
    { label: "ФИО", start: 0, end: 4 },
    { label: "Полный адрес", start: 13, end: 19 },
  ];
  assert.equal(buildObfuscated(text, spans), "[ФИО] живёт в [ПОЛНЫЙ_АДРЕС]");
});

test("buildObfuscated sorts spans and leaves text without spans unchanged", () => {
  assert.equal(buildObfuscated("hello", []), "hello");
  const text = "Иван живёт в Москве";
  const spans = [
    { label: "Полный адрес", start: 13, end: 19 }, // deliberately before the ФИО span
    { label: "ФИО", start: 0, end: 4 },
  ];
  assert.equal(buildObfuscated(text, spans), "[ФИО] живёт в [ПОЛНЫЙ_АДРЕС]");
});

test("buildHighlighted wraps entities (with label title) and escapes the rest", () => {
  const text = "Иван <b> Москва";
  const spans = [
    { label: "ФИО", start: 0, end: 4 },
    { label: "Полный адрес", start: 9, end: 15 },
  ];
  assert.equal(
    buildHighlighted(text, spans),
    '<span class="entity" title="ФИО">Иван</span> &lt;b&gt; ' +
      '<span class="entity" title="Полный адрес">Москва</span>'
  );
});

test("escapeHtml neutralises all five HTML-significant chars", () => {
  assert.equal(
    escapeHtml(`<a href="x" data='y'>&`),
    "&lt;a href=&quot;x&quot; data=&#39;y&#39;&gt;&amp;"
  );
});

test("obfuscation and highlight derive from the SAME spans", () => {
  // The contract that makes the demo safe: what is hidden in the bubble is exactly
  // what is replaced in the outgoing text.
  const text = "Меня зовут Анна Петрова, телефон 89991234567";
  const tokens = [
    tok("ФИО", 11, 15), // "Анна"
    tok("ФИО", 16, 23), // "Петрова" (adjacent -> one span)
    tok("Номер телефона", 33, 44), // "89991234567"
  ];
  const spans = mergeSpans(tokens);
  assert.equal(buildObfuscated(text, spans), "Меня зовут [ФИО], телефон [НОМЕР_ТЕЛЕФОНА]");
  assert.match(buildHighlighted(text, spans), /class="entity" title="ФИО">Анна Петрова</);
});
