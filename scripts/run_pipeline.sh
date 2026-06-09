#!/usr/bin/env bash
# End-to-end PII NER pipeline: train two base models in parallel on separate GPUs
# (each with k-fold CV + uncertainty mining + augmented retrain), then blend every
# resulting checkpoint into a single submission.
set -euo pipefail

RUBERT="DeepPavlov/rubert-base-cased"
CONV="Data-Lab/rubert-base-cased-conversational_ner-v1"
FOLDS=3

RUBERT_TAG="rubert-base-cased"
CONV_TAG="rubert-base-cased-conversational_ner-v1"

# --- 1. Train both models in parallel (mining + augmented retrain included) -----
echo "[pipeline] Training $RUBERT on cuda:0 and $CONV on cuda:1..."
CUDA_VISIBLE_DEVICES=0 uv run pii-train --model "$RUBERT" --folds "$FOLDS" --mine \
    2>&1 | sed 's/^/[rubert] /' &
PID_RUBERT=$!
CUDA_VISIBLE_DEVICES=1 uv run pii-train --model "$CONV" --folds "$FOLDS" --mine \
    2>&1 | sed 's/^/[conv]   /' &
PID_CONV=$!

wait $PID_RUBERT || { echo "[pipeline] rubert training FAILED"; exit 1; }
echo "[pipeline] rubert training done."
wait $PID_CONV   || { echo "[pipeline] conv training FAILED";   exit 1; }
echo "[pipeline] conv training done."

# --- 2. Collect every checkpoint (folds + augmented, both models) ---------------
ALL_DIRS=""
for i in $(seq 1 "$FOLDS"); do ALL_DIRS="$ALL_DIRS ner_model_${RUBERT_TAG}_fold${i}"; done
ALL_DIRS="$ALL_DIRS ner_model_${RUBERT_TAG}_augmented"
for i in $(seq 1 "$FOLDS"); do ALL_DIRS="$ALL_DIRS ner_model_${CONV_TAG}_fold${i}"; done
ALL_DIRS="$ALL_DIRS ner_model_${CONV_TAG}_augmented"

# --- 3. (Optional) OOF sanity check on the rubert folds -------------------------
RUBERT_FOLDS=""
for i in $(seq 1 "$FOLDS"); do RUBERT_FOLDS="$RUBERT_FOLDS ner_model_${RUBERT_TAG}_fold${i}"; done
echo "[pipeline] OOF evaluation of rubert folds..."
uv run pii-evaluate --model-dirs $RUBERT_FOLDS --oof --folds "$FOLDS" --per-label \
    2>&1 | sed 's/^/[eval]   /'

# --- 4. Blend everything into the submission ------------------------------------
echo "[pipeline] Blending all checkpoints..."
uv run pii-blend --model-dirs $ALL_DIRS --output submission.csv 2>&1 | sed 's/^/[blend]  /'
echo "[pipeline] Done. Submission saved to submission.csv"
