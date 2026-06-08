# -*- coding: utf-8 -*-
"""
Convert the trained ModernBERT token-classification model to ONNX (int8) for
in-browser inference with transformers.js, and lay it out the way the Hub /
transformers.js expects:

    web_model/
        config.json
        tokenizer.json
        tokenizer_config.json
        special_tokens_map.json      (if present)
        onnx/
            model.onnx               (fp32)
            model_quantized.onnx     (int8 — what transformers.js loads by default)

This step REQUIRES network + extra packages that are NOT in the training venv:

    pip install "optimum[onnxruntime]>=1.24"

(ModernBERT export support landed in optimum 1.24.) Then:

    python convert_to_onnx.py

Afterwards upload the whole `web_model/` folder to a Hugging Face Hub repo and
point MODEL_ID in docs/index.html at it. See README → "Деплой на GitHub Pages".
"""
from __future__ import annotations

import os
import shutil

SRC = "ner_model_final"
OUT = "web_model"
ONNX_DIR = os.path.join(OUT, "onnx")


def main() -> None:
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        raise SystemExit(
            "optimum is not installed.\n"
            '  pip install "optimum[onnxruntime]>=1.24"\n'
            "then re-run this script in an environment with network access."
        )

    os.makedirs(ONNX_DIR, exist_ok=True)

    # 1) Export fp32 ONNX + copy config/tokenizer into OUT (optimum does both).
    print(f"Exporting ONNX from {SRC} -> {OUT} ...")
    main_export(
        model_name_or_path=SRC,
        output=OUT,
        task="token-classification",
        opset=14,
    )

    # optimum writes model.onnx at the OUT root; transformers.js wants it under onnx/.
    root_model = os.path.join(OUT, "model.onnx")
    fp32 = os.path.join(ONNX_DIR, "model.onnx")
    if os.path.exists(root_model):
        shutil.move(root_model, fp32)
    if not os.path.exists(fp32):
        raise SystemExit("ONNX export did not produce model.onnx")
    print(f"  fp32 size: {os.path.getsize(fp32) / 1e6:.1f} MB")

    # 2) Dynamic int8 quantization -> model_quantized.onnx (browser-sized).
    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8 = os.path.join(ONNX_DIR, "model_quantized.onnx")
    print("Quantizing int8 ...")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    print(f"  int8 size: {os.path.getsize(int8) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
