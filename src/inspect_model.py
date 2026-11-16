#!/usr/bin/env python3
"""
Inspect a TFLite detector artifact: SHA-256, size, input/output tensor
specs, quantization params, and an optional host-CPU latency benchmark.

Read-only: never writes to the model path. Works with ai-edge-litert,
tflite-runtime, or tensorflow (whichever is installed).

Usage:
    python src/inspect_model.py models/artifacts/yolo11n-baseline.tflite [--bench 20]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def load_interpreter(path: str):
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=path), "ai-edge-litert"
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=path), "tflite-runtime"
    except ImportError:
        pass
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "No TFLite interpreter is installed. Install ai-edge-litert, "
            "tflite-runtime, or tensorflow to inspect this model.") from exc
    return tf.lite.Interpreter(model_path=path), "tensorflow"


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--bench", type=non_negative_int, default=0,
                    help="latency runs (0 = skip)")
    args = ap.parse_args()

    p = Path(args.model)
    if not p.is_file():
        print(f"model not found or not a regular file: {p}", file=sys.stderr)
        return 1

    try:
        data = p.read_bytes()
    except OSError as exc:
        print(f"cannot read model {p}: {exc}", file=sys.stderr)
        return 1
    if not data:
        print(f"model is empty: {p}", file=sys.stderr)
        return 1
    report = {
        "file": str(p),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

    try:
        interp, backend = load_interpreter(str(p))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"model load failed: {exc}", file=sys.stderr)
        return 2
    report["interpreter"] = backend
    try:
        interp.allocate_tensors()

        def spec(d):
            q = d["quantization"]
            return {
                "name": d["name"],
                "shape": [int(s) for s in d["shape"]],
                "dtype": str(d["dtype"]),
                "quantization": {"scale": float(q[0]), "zero_point": int(q[1])},
            }

        report["inputs"] = [spec(d) for d in interp.get_input_details()]
        report["outputs"] = [spec(d) for d in interp.get_output_details()]
        report["tensor_count"] = len(interp.get_tensor_details())

        if args.bench > 0:
            import numpy as np
            inp = interp.get_input_details()[0]
            x = np.random.rand(*inp["shape"]).astype(inp["dtype"])
            interp.set_tensor(inp["index"], x)
            interp.invoke()  # warmup
            times = []
            for _ in range(args.bench):
                t = time.perf_counter()
                interp.set_tensor(inp["index"], x)
                interp.invoke()
                times.append((time.perf_counter() - t) * 1000)
            times.sort()
            report["host_latency_ms"] = {
                "median": round(times[len(times) // 2], 1),
                "min": round(times[0], 1),
                "max": round(times[-1], 1),
                "runs": args.bench,
                "note": "host CPU only — NOT an Android measurement",
            }
    except Exception as exc:
        print(f"model inspection failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
