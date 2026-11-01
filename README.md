# BASERA Vision Lab

Independent computer-vision R&D project: a specialized object & hazard
detector for blind and visually impaired users, designed to run **offline
on Android phones** (TFLite / LiteRT), as a future model candidate for the
BASERA assistive app.

## Hard boundaries

- This repo is **fully separate** from the BASERA mobile app.
- It never modifies the app repo, Expo, or Supabase, never publishes an
  APK/OTA, and never replaces the app's current YOLO11n model.
- A model produced here becomes a *candidate* only after it passes the
  evaluation and on-device compatibility gates in the Phase 1 report.
- Datasets and large weights are never committed to git (see
  `data/README.md`, `models/README.md`).

## Status

Phase 1 (inspection & evidence-based plan) — complete, awaiting owner
approval of the decision gates before any training spend.
See `docs/PHASE1_REPORT.md`.

## Layout

- `docs/` — reports, dataset/license reviews, plans
- `configs/` — verified baseline spec, proposed class taxonomy
- `src/` — inspection/eval tooling (no app code)
- `models/` — registry + (gitignored) local artifacts
- `data/` — dataset policy + manifests (no images)
- `experiments/` — training run records (configs, logs, metrics)
