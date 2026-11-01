# GitHub linking & Phase 2 status — evidence log

Date: 2026-11-01 (Africa/Cairo)

## GitHub repository

- URL: https://github.com/0xsabry/basera-vision-lab — owner `0xsabry`,
  **Private**, default branch `main`.
- Creation path record (for future reference):
  1. Muse GitHub App integration `create_repository` → **failed twice**
     with `403 Resource not accessible by integration` on
     `POST /user/repos`. GitHub Apps cannot create user repositories via
     that endpoint — a platform limitation of app tokens, not an account
     permission problem.
  2. `gh` CLI initially unauthenticated; no GitHub tokens present in the
     environment (checked by name only).
  3. **Successful path:** browser session → GitHub web UI. The owner
     signed in (saved login + 2FA via GitHub Mobile) and the empty
     private repo was created via github.com/new (no README/.gitignore/
     license).
  4. Push path: GitHub CLI device authorization (`gh auth login --web`),
     approved by the owner, then a real `git push` — preserving the
     local commit history and SHAs exactly.
- Note: the Muse GitHub App (connector) still cannot *see* this repo
  (its API reads return 404) until the owner adds `basera-vision-lab`
  to the app's repository selection in GitHub settings. Git operations
  via `gh`/git are unaffected.

## Pre-push audit (executed before pushing)

- Tracked files: **18**, total **70,197 bytes**; largest file 23,409 B
  (Phase 1 report). No datasets, no images, no credentials.
- Secret scan over tracked content (token/key/secret/password/Bearer/
  private-key patterns): **no credential material found**.
- `.gitignore` verified active: `models/artifacts/` (baseline TFLite +
  ONNX, ~21 MB) shows as ignored (`!!`) and is in **no** commit.
- Commits pushed:
  - `0f8a111d4675c0eba64e5c5b2d8ac44dcc49295f` — Phase 1
  - `16723ede9a27541eed7f23a16e5f15d66b1ebbe4` — Phase 2 design + pilot prep
- Push verification: `git ls-remote origin` returned
  `16723ede…` for both `HEAD` and `refs/heads/main`, **identical** to the
  local HEAD at push time. GitHub API confirms `private: true`,
  `default_branch: main`, and `docs/` contents present.

## Independence check

No file in the BASERA app repo, Expo/EAS, or Supabase was modified by
this project. All Vision Lab work lives in this repository only.

## Phase 2 deliverables (in this repo)

- `docs/DATASET_ARCHITECTURE_V2.md` — composition targets, group splits,
  leakage rules (group split + pHash), metadata schema, condition quotas,
  versioning + dataset-card requirements.
- `docs/CAPTURE_PROTOCOL_EGYPT.md` — capturer safety, Egypt sensitivity
  rules (no security/military subjects), privacy & consent, technique,
  session discipline, priority shot list, ingest checklist.
- `docs/ANNOTATION_GUIDELINES.md` — universal box rules, per-class edge
  cases, image-level tags, QA gates (double annotation, IoU ≥ 0.85,
  class agreement ≥ 95%, 200-image gold set).
- `configs/classes_v2.yaml` — frozen class IDs (29 classes, 3 tiers) +
  8-class pilot subset.
- `pilot/` — Kaggle pilot, prepared and syntax-verified, **$0, not yet
  executed**: `run_pilot.py` (smoke/prep/train/report), `kaggle_pilot.ipynb`,
  README with success criteria and the post-pilot budget gate.
- No training has run. No money has been spent. The $75 ceiling stands;
  pilot results will recalibrate the Phase 3 matrix before any spend.
