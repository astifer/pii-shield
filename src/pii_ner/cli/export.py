"""Export a model to ONNX int8 for the browser demo (transformers.js + WebGPU).

Examples:
    pii-export-onnx --scaffold                       # untrained head, demo runs now
    pii-export-onnx --model-dir ner_model_rubert_augmented
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pii_ner.export_onnx import DEFAULT_OUT, export


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export NER model to ONNX int8 for the browser")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model-dir", help="Trained ner_model_* checkpoint dir to export")
    g.add_argument("--scaffold", action="store_true",
                   help="Build an untrained 59-BIO head on the base rubert model")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"Output model folder (default: {DEFAULT_OUT})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export(model_dir=args.model_dir, scaffold=args.scaffold, out=Path(args.out))


if __name__ == "__main__":
    main()
