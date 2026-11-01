# Kaggle Pilot — prepared, $0, not yet executed

## Why a pilot first

Phase 1's training times and costs are estimates. This pilot replaces
them with **measured** numbers on a free Kaggle GPU (T4/P100) before any
paid run, any full dataset freeze, or any model decision.

## What it proves (success criteria)

1. Both candidate weights (`yolo11n.pt`, `yolo26n.pt`) download from
   Ultralytics and train under the same config.
2. Open Images subset → YOLO-format dataset builds cleanly for the 8
   pilot classes (all shared COCO/OID classes; no custom capture needed).
3. Per-epoch time is measured → full-run GPU hours and dollars become a
   calculation, not a guess.
4. TFLite FP32 + FP16 exports succeed and file sizes/metrics are recorded
   next to the PyTorch metrics (export-parity check).
5. `runs/pilot/REPORT.md` is produced and committed (numbers only —
   weights stay out of git).

Explicitly **not** proven by the pilot: anything about Egyptian classes,
night performance, or the real Dataset v1. Pilot metrics must never be
quoted as model quality.

## How to run (Kaggle, free tier)

1. Create a Kaggle notebook, accelerator = GPU (T4/P100), Internet = on.
2. Add this repo's `pilot/`, `src/`, `configs/` as input (Kaggle Dataset
   upload of the repo zip, or git clone once the GitHub repo is linked —
   private repo clone needs a Kaggle secret with a read-only token;
   the zip route needs no credentials and is preferred for the pilot).
3. Open `kaggle_pilot.ipynb` (or run the commands):
   `pip install ultralytics fiftyone ai-edge-litert` →
   `run_pilot.py --stage smoke` → `--stage prep` →
   `--stage train --model yolo11n` → `--stage train --model yolo26n` →
   `--stage report`.
4. Expected wall time [Phase-1 estimate]: smoke ~15 min, prep ~30–60 min
   (download-bound), each train ~1.5–4 h at 30 epochs on T4 — inside one
   12 h session and the ~30 h/week free quota.
5. Download `runs/pilot/REPORT.md` + both `results.json` and commit them
   under `experiments/pilot-001/`.

## Gate after the pilot

With measured epoch times, recompute the Phase 3 experiment matrix cost.
If the projected total exceeds the approved $75 ceiling, the matrix is
cut (fewer runs / fewer epochs) **before** spending — never mid-run.
