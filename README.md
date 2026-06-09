# PII NER + Browser Shield

Russian **PII named-entity recognition** for the *llm-march2026-alfabank* ("PII Shield")
competition. Given a Russian text, predict character spans `[(start, end, label), ...]`
over **29 PII entity types** (ФИО, Номер карты, СНИЛС, паспортные данные, …).

This repo has two halves:

1. **Training pipeline** (`src/pii_ner/`) — a probability-blended ensemble of
   token-classification models (`DeepPavlov/rubert-base-cased` and a conversational
   ruBERT) with an uncertainty-mining self-training loop.
2. **Browser demo** (`web/` + `pii_ner.web`) — runs a trained model **on-device via
   WebGPU** (transformers.js), obfuscates every detected PII span (`[ФИО]`,
   `[ПОЛНЫЙ_АДРЕС]`, …) **before** the text is sent to an LLM. The backend LLM is a
   FastAPI mock that echoes the obfuscated message, so raw PII never leaves the browser.

## Install

Uses [`uv`](https://docs.astral.sh/uv/). The package installs on first `uv run`:

```bash
uv sync                                   # training deps only
uv sync --extra web --extra export --extra dev   # + browser demo, ONNX export, tests
```

## Data

Place the competition files under `data/` (gitignored):

- `data/train_dataset.tsv` — columns `text` / `target` / `entity`, where `target` is a
  stringified `[(start, end, label), ...]`.
- `data/private_test_dataset.csv` — columns `id` / `text`.

## Training usage

Console commands (run via `uv run`):

```bash
# Train k-fold models (resumable — skips folds with an existing checkpoint)
uv run pii-train --model DeepPavlov/rubert-base-cased --folds 3

# ...plus uncertainty mining + augmented retrain on confident pseudo-labels
uv run pii-train --model DeepPavlov/rubert-base-cased --folds 3 --mine

# Out-of-fold span-level P/R/F1 with a per-label breakdown
uv run pii-evaluate --model-dirs ner_model_rubert-base-cased_fold{1,2,3} --oof --folds 3 --per-label

# Blend any set of checkpoints into submission.csv
uv run pii-blend --model-dirs ner_model_rubert-base-cased_fold1 ner_model_rubert-base-cased_augmented \
                 --output submission.csv
```

Full train → mine → evaluate → blend flow across two GPUs: `./scripts/run_pipeline.sh`.

**Key invariant:** models are ensembled by averaging their **raw softmax arrays**, so
every caller decodes through the same functions in `inference.py` (`MAX_LENGTH=512`,
`THRESHOLD=0.6`). Don't fork that logic.

## Browser demo (WebGPU)

```bash
uv sync --extra web --extra export

# 1. Export a model to ONNX int8 for transformers.js.
#    Scaffold (untrained head) — the demo runs immediately, predictions are random:
uv run pii-export-onnx --scaffold
#    Or a REAL trained checkpoint (real entity detection, no frontend changes):
uv run pii-export-onnx --model-dir ner_model_rubert-base-cased_augmented

# 2. Serve it.
uv run pii-serve            # http://127.0.0.1:8000/
```

What you get: a one-time model **download progress bar** → on-device NER → per-message
**inference time in ms**, with detected entities **highlighted in red** and replaced by
their `[TAG]` in the message sent to the mock LLM.

- `pii-export-onnx` writes `web/models/pii-ner-rubert/onnx/model_quantized.onnx` (int8,
  with an fp32 fallback) plus the tokenizer/config, and regenerates
  `web/labels.generated.js` (the entity→tag map) from `pii_ner.labels` so the frontend
  can never drift from the trained label set.
- The mock LLM is `pii_ner.web.server.mock_llm` — swap in a real call there; the contract
  (obfuscated text in, HTML fragment out) is unchanged.

## Tests & benchmark

```bash
uv run pytest tests/                       # FastAPI backend + label-tag coverage
node --test web/obfuscate.test.js          # pure entity logic (merge/highlight/obfuscate)
uv run python benchmark_onnx.py            # ONNX inference latency + throughput (int8 vs fp32)
```

`benchmark_onnx.py` reports per-sentence latency percentiles and tokens/sec on CPU
onnxruntime (server-side analog of the browser's "ms" readout; the demo itself uses
WebGPU). Use it for regression tracking and quantization comparison.

## Architecture

```
src/pii_ner/
  labels.py          BIO label schema — single source of truth for label<->id (29 types)
  config.py          paths, decoding thresholds, training hyperparameters
  data.py            TSV/CSV loading, char-span -> BIO token alignment, NERDataset
  inference.py       softmax extraction, probability blending, BIO decoding
  metrics.py         token-level (Trainer) and span-level (eval) P/R/F1
  mining.py          OOF / blended uncertainty mining
  training.py        per-fold fine-tuning + pseudo-label augmented retraining
  postprocessing.py  submission span cleanup (boundary expansion, overlaps, …)
  ema.py             optional EMA-smoothed weights (`--ema`)
  export_onnx.py     export a checkpoint -> ONNX int8 for the browser + labels.generated.js
  cli/               thin argparse entry points (train/evaluate/blend/mine/export/serve)
  web/server.py      FastAPI mock-LLM backend + static serving
web/                 transformers.js + WebGPU frontend (index.html, app.js, obfuscate.js)
```

## Notes

- Checkpoints (`ner_model_*/`), data, pipeline CSVs, and the exported `web/models/` are
  gitignored — build artifacts, regenerated by the commands above.
- Scaffold export uses an **untrained** classification head: the architecture, tokenizer
  and 59-BIO label set match a real checkpoint, so swapping in a trained model
  (`pii-export-onnx --model-dir …`) needs no frontend or test changes — only *which*
  tokens fire becomes meaningful.
