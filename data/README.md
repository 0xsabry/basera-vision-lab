# Data policy

Datasets are **never** committed to git and never uploaded to GitHub.

- Git tracks only: dataset cards (`docs/` / `data/*.md` manifests), class
  taxonomy, annotation guidelines, split manifests (file lists + hashes),
  and download/prep scripts.
- Raw images live in external storage (Kaggle Datasets / cloud bucket /
  local disk), referenced by a dataset version ID (e.g. `bvl-ds-v1`).
- Every source must have a recorded license verdict in
  `docs/DATASETS_AND_LICENSES.md` **before** its images enter any split.
- No personal data without consent. Egyptian street captures follow the
  capture protocol in the Phase 2 plan (faces/plates blurred in stored
  copies, no children-focused capture, consent for identifiable subjects
  where feasible).
- Splits are by **capture session / location / video**, never by random
  frame, to prevent near-duplicate leakage between train/val/test.
