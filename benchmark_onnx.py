"""Performance benchmark for the exported NER ONNX model.

Measures inference latency + throughput of the int8 model (and the fp32 model, if
present) with onnxruntime on CPU -- the server-side analog of the in-browser "ms"
readout. It reports per-sentence latency percentiles and tokens/sec so you can track
regressions and compare quantizations.

Run:
    uv sync --extra export                     # transformers, onnxruntime
    uv run pii-export-onnx --scaffold          # produces the model first
    uv run python benchmark_onnx.py [--iters 50 --warmup 5]
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).parent / "web" / "models" / "pii-ner-rubert"
ONNX_DIR = MODEL_DIR / "onnx"

# Representative Russian inputs of increasing length (the kind a user would type).
SAMPLES = [
    "Привет",
    "Меня зовут Анна Петрова.",
    "Я живу в Москве на Тверской улице, дом 5.",
    "Здравствуйте! Меня зовут Иван Сергеевич Кузнецов, я проживаю по адресу "
    "город Санкт-Петербург, Невский проспект, дом 12, квартира 34.",
    "Перешлите документы Марии Ивановой и Петру Смирнову; адреса: Ленина 1 в "
    "Новосибирске и Гагарина 7 в Казани. " * 3,
]


def humansize(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def bench_model(name: str, onnx_path: Path, tokenizer, iters: int, warmup: int):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}

    # Pre-tokenise each sample and keep only the inputs this graph declares.
    feeds = []
    seq_lens = []
    for text in SAMPLES:
        enc = tokenizer(text, return_tensors="np")
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in input_names}
        feeds.append(feed)
        seq_lens.append(int(enc["input_ids"].shape[1]))

    # Warmup (JIT/alloc) on the longest sample.
    longest = feeds[int(np.argmax(seq_lens))]
    for _ in range(warmup):
        sess.run(None, longest)

    # Timed runs, round-robin over the samples.
    latencies_ms = []
    tokens_done = 0
    for i in range(iters):
        feed = feeds[i % len(feeds)]
        t0 = time.perf_counter()
        sess.run(None, feed)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        tokens_done += seq_lens[i % len(feeds)]

    lat = sorted(latencies_ms)
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
    total_s = sum(latencies_ms) / 1000
    print(f"\n=== {name}  ({humansize(onnx_path)}) ===")
    print(f"  providers : {sess.get_providers()[0]}")
    print(f"  seq lens  : {seq_lens} tokens")
    print(f"  iters     : {iters} (warmup {warmup})")
    print(f"  latency ms: mean {statistics.mean(latencies_ms):6.1f} | "
          f"p50 {p(0.50):6.1f} | p95 {p(0.95):6.1f} | "
          f"min {lat[0]:6.1f} | max {lat[-1]:6.1f}")
    print(f"  throughput: {iters / total_s:6.1f} sent/s | "
          f"{tokens_done / total_s:7.0f} tokens/s")
    return statistics.mean(latencies_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if not MODEL_DIR.exists():
        raise SystemExit(f"Model not found at {MODEL_DIR}. Run `pii-export-onnx --scaffold` first.")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    targets = [("int8 (q8)", ONNX_DIR / "model_quantized.onnx")]
    if (ONNX_DIR / "model.onnx").exists():
        targets.append(("fp32", ONNX_DIR / "model.onnx"))

    means = {}
    for name, path in targets:
        if path.exists():
            means[name] = bench_model(name, path, tokenizer, args.iters, args.warmup)
        else:
            print(f"\n(skipping {name}: {path} not found)")

    if "int8 (q8)" in means and "fp32" in means:
        speedup = means["fp32"] / means["int8 (q8)"]
        print(f"\nint8 vs fp32 mean-latency speedup: {speedup:.2f}x")
    print("\nNote: browser numbers differ -- this is CPU onnxruntime; the demo runs "
          "WebGPU. Use this for regression tracking and quantization comparison.")


if __name__ == "__main__":
    main()
