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


class InterpreterUnavailable(RuntimeError):
    """Raised only when no supported TFLite backend is installed."""


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
        raise InterpreterUnavailable(
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
    except InterpreterUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print(
            "model load failed: the model file or installed TFLite runtime "
            "could not be loaded.",
            file=sys.stderr,
        )
        return 2

    try:
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        tensor_details = interp.get_tensor_details()
        if not input_details or not output_details or not tensor_details:
            raise ValueError("required tensor metadata is empty")

        def spec(detail):
            quantization = detail["quantization"]
            if len(quantization) != 2:
                raise ValueError("invalid quantization metadata")
            return {
                "name": detail["name"],
                "shape": [int(size) for size in detail["shape"]],
                "dtype": str(detail["dtype"]),
                "quantization": {
                    "scale": float(quantization[0]),
                    "zero_point": int(quantization[1]),
                },
            }

        report["interpreter"] = backend
        report["inputs"] = [spec(detail) for detail in input_details]
        report["outputs"] = [spec(detail) for detail in output_details]
        report["tensor_count"] = len(tensor_details)
    except Exception:
        print(
            "model inspection failed: the interpreter could not allocate "
            "tensors or returned invalid metadata.",
            file=sys.stderr,
        )
        return 2

    if args.bench > 0:
        try:
            import numpy as np

            inp = input_details[0]
            x = np.random.rand(*inp["shape"]).astype(inp["dtype"])
            interp.set_tensor(inp["index"], x)
            interp.invoke()  # warmup
            times = []
            for _ in range(args.bench):
                started = time.perf_counter()
                interp.set_tensor(inp["index"], x)
                interp.invoke()
                times.append((time.perf_counter() - started) * 1000)
            times.sort()
            report["host_latency_ms"] = {
                "median": round(times[len(times) // 2], 1),
                "min": round(times[0], 1),
                "max": round(times[-1], 1),
                "runs": args.bench,
                "note": "host CPU only — NOT an Android measurement",
            }
        except Exception:
            print(
                "benchmark failed: the interpreter could not run the "
                "requested host-CPU measurement.",
                file=sys.stderr,
            )
            return 2
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
