#!/usr/bin/env python3
"""
BASERA Vision Lab — Kaggle Pilot runner ($0, free GPU tier).

Purpose: validate the END-TO-END pipeline and CALIBRATE real training
costs before any paid run or full dataset work. NOT a production run.

Stages:
  smoke : coco8, 10 epochs, YOLO11n + YOLO26n — pipeline integrity only.
  prep  : build the pilot dataset (~3,000 images, 8 public classes) from
          Open Images V7 via FiftyOne (annotations CC BY 4.0 / images
          CC BY 2.0 — see docs/DATASETS_AND_LICENSES.md).
  train : fine-tune one model (--model yolo11n|yolo26n) for 30 epochs on
          the pilot dataset; records per-epoch time, val metrics, exports
          TFLite FP32 + FP16, and writes runs/pilot/<model>/results.json.
  report: aggregate all results.json into runs/pilot/REPORT.md.

Pilot classes (contiguous pilot ids -> taxonomy v2 ids):
  0 person(0) 1 car(1) 2 bus(2) 3 truck(3) 4 motorcycle(4)
  5 bicycle(5) 6 dog(7) 7 traffic_light(8)
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "pilot"
DS_DIR = PILOT / "pilot_ds"
RUNS = ROOT / "runs" / "pilot"

PILOT_CLASSES = ["person", "car", "bus", "truck", "motorcycle",
                 "bicycle", "dog", "traffic_light"]
OID_LABELS = ["Person", "Car", "Bus", "Truck", "Motorcycle",
              "Bicycle", "Dog", "Traffic light"]
MODELS = {"yolo11n": "yolo11n.pt", "yolo26n": "yolo26n.pt"}


def stage_smoke() -> None:
    from ultralytics import YOLO
    for name, weights in MODELS.items():
        print(f"[smoke] {name} on coco8, 10 epochs")
        m = YOLO(weights)
        m.train(data="coco8.yaml", epochs=10, imgsz=640,
                project=str(RUNS), name=f"smoke-{name}", exist_ok=True)
        m.export(format="tflite", imgsz=640)
    print("[smoke] OK — pipeline integrity confirmed for both models")


def stage_prep(max_samples: int = 3000) -> None:
    import fiftyone as fo

    print(f"[prep] downloading Open Images V7 subset: {OID_LABELS}")
    ds = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=OID_LABELS,
        max_samples=max_samples,
        dataset_name="bvl-pilot-oid",
        overwrite=True,
    )
    # Split by image (public web photos have no capture groups; the real
    # dataset uses group splits — pilot is pipeline validation only).
    import random
    random.seed(42)
    all_ids = [s.id for s in ds]
    random.shuffle(all_ids)
    train_ids = all_ids[: int(len(all_ids) * 0.85)]
    views = {}
    for name, view in (("train", ds.select(train_ids)),
                       ("val", ds.exclude(train_ids))):
        view.export(
            export_dir=str(DS_DIR / name),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=PILOT_CLASSES,
        )
        views[name] = len(view)
    yaml = PILOT / "pilot_data.yaml"
    yaml.write_text(
        f"path: {DS_DIR}\ntrain: train/images\nval: val/images\n"
        f"names: {PILOT_CLASSES}\n")
    print(f"[prep] wrote {yaml} — sizes: {views}")


def stage_train(model_name: str, epochs: int = 30) -> None:
    from ultralytics import YOLO

    if model_name not in MODELS:
        raise SystemExit(f"unknown model {model_name}; choose {list(MODELS)}")
    data_yaml = PILOT / "pilot_data.yaml"
    if not data_yaml.exists():
        raise SystemExit("pilot_data.yaml missing — run --stage prep first")

    epoch_times = []
    state = {"t0": None}

    def on_epoch_start(_):
        state["t0"] = time.perf_counter()

    def on_epoch_end(_):
        epoch_times.append(round(time.perf_counter() - state["t0"], 2))

    m = YOLO(MODELS[model_name])
    m.add_callback("on_train_epoch_start", on_epoch_start)
    m.add_callback("on_train_epoch_end", on_epoch_end)
    t_start = time.perf_counter()
    m.train(data=str(data_yaml), epochs=epochs, imgsz=640,
            project=str(RUNS), name=f"train-{model_name}", exist_ok=True)
    total_s = time.perf_counter() - t_start

    metrics = m.val(data=str(data_yaml))
    exports = {}
    for half in (False, True):
        path = m.export(format="tflite", imgsz=640, half=half)
        exports["fp16" if half else "fp32"] = {
            "path": str(path), "bytes": Path(path).stat().st_size}

    out = RUNS / f"train-{model_name}"
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "model": model_name,
        "weights": MODELS[model_name],
        "epochs": epochs,
        "epoch_time_s": epoch_times,
        "epoch_time_median_s": sorted(epoch_times)[len(epoch_times) // 2]
        if epoch_times else None,
        "total_train_s": round(total_s, 1),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "exports_tflite": exports,
        "note": "PILOT numbers on a public-data subset. Not comparable "
                "to Dataset v1 results; purpose is calibration only.",
    }
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def stage_report() -> None:
    rows = []
    for f in sorted(RUNS.glob("train-*/results.json")):
        r = json.loads(f.read_text())
        rows.append(
            f"| {r['model']} | {r['epochs']} | {r['epoch_time_median_s']}s | "
            f"{r['total_train_s']}s | {r['map50']:.3f} | "
            f"{r['map50_95']:.3f} | {r['recall']:.3f} | "
            f"{r['exports_tflite']['fp32']['bytes']} | "
            f"{r['exports_tflite']['fp16']['bytes']} |")
    md = ("# Pilot report (calibration only)\n\n"
          "| model | epochs | epoch median | total train | mAP50 | "
          "mAP50-95 | recall | TFLite FP32 B | TFLite FP16 B |\n"
          "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
          "Extrapolation rule for the full-run budget: "
          "full_epoch_s ≈ epoch_median × (dataset_v1_images / pilot_images), "
          "then × epochs × runs, priced at the chosen cloud GPU rate. "
          "Replace the Phase 1 estimates with these measured numbers.\n")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "REPORT.md").write_text(md)
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "prep", "train", "report"])
    ap.add_argument("--model", default="yolo11n")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-samples", type=int, default=3000)
    a = ap.parse_args()
    if a.stage == "smoke":
        stage_smoke()
    elif a.stage == "prep":
        stage_prep(a.max_samples)
    elif a.stage == "train":
        stage_train(a.model, a.epochs)
    else:
        stage_report()


if __name__ == "__main__":
    main()
