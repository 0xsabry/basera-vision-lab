# Dataset Architecture v2 — BASERA Vision Lab

Status: design for **Dataset v1 (`bvl-ds-v1`)**. No images are in git.
Class IDs: `configs/classes_v2.yaml` (frozen).

## 1. Design goals

1. Optimize for the failure that matters: a **missed hazard** for a walking
   blind user — not leaderboard mAP.
2. Match the real input distribution: handheld phone photos/video taken
   while walking in Egypt — motion blur, bad framing, low light, tilted
   angles — not clean web photography.
3. License-clean by construction: every image carries a source + license
   record; a run is "commercial-clean" only if every source in it is.
4. Zero leakage: splits that cannot share near-duplicates.

## 2. Composition targets (v1)

| Split | Images (target) | Role |
|---|---|---|
| Train | ~24,000 | fine-tuning |
| Val | ~3,500 | model/epoch selection, threshold tuning |
| Test | ~3,500 | **frozen**, touched only for final evaluation reports |

Per-class instance minimums live in `classes_v2.yaml` (Tier A 1,000–3,000,
Tier B 600–1,200, Tier C 800). A class that misses its minimum at freeze
time is either merged (documented) or dropped from v1 — never padded with
near-duplicates.

**Sources per class** (details + license verdicts in
`DATASETS_AND_LICENSES.md`):
- COCO/Open Images subsets → shared Tier A/B classes (bootstrap volume).
- Custom Egyptian capture → all Egypt-specific classes (`tuk_tuk`,
  `vendor_cart`), most of `pothole`, `stairs`, `pole`, `barrier`,
  `crosswalk`, all Tier C, plus Egyptian-context instances of every
  shared class (target: ≥40% of Tier A instances are Egyptian-context).
- Hard negatives ≥15% of train images (see annotation guidelines).

**Condition quotas (train, measured and reported at freeze):**
daylight ≥40% · dusk/night street-light ≥20% · low-light indoor ≥10% ·
backlight ≥5% · visible motion blur ≥10% · occlusion/crowding ≥20% of
images containing at least one occluded instance.

## 3. Storage layout (outside git)

```
<storage>/bvl-ds-v1/
  raw/            # untouched originals per source (read-only after ingest)
  work/           # de-duplicated, privacy-processed working copies
  splits/
    train/images|labels   val/...   test/...
  manifests/
    images.parquet        # one row per image (schema §4)
    splits.json           # group -> split assignment + hash of file lists
    sources.md            # per-source license evidence
  qa/                     # double-annotation samples, agreement reports
```

Storage candidates (choose at ingest time, record in the dataset card):
Kaggle Datasets (private), Hugging Face Datasets (private), or a cloud
bucket. Git stores only manifests' hashes + the dataset card.

## 4. Per-image metadata schema (manifest)

`image_id, file_name, sha256, width, height, source_id, source_license,
split, group_id, capture_session_id, city, context (street/indoor/transit),
lighting (day/dusk/night/low/backlight), blur (none/mild/strong),
device, has_faces (bool, blurred in work copy), annotator, reviewer,
instances: [{class_id, bbox_xywhn, occluded, truncated, difficult}]`

`group_id` = capture session × location (custom) or source-video /
source-collection id (public data). **The split is assigned per group,
never per image.**

## 5. Leakage prevention (hard rules)

1. Group-level split (§4): all images of a group land in exactly one split.
2. Perceptual-hash (pHash) dedupe across the whole corpus at ingest;
   any cross-split near-duplicate pair found later quarantines both
   images and triggers a manifest note.
3. Video frames: max 1 frame per 2 s per scene, and a scene stays in one
   group.
4. Test set is frozen at v1 freeze: no image moves out of test into
   train in any later patch version (v1.1+ may only *add* new groups).

## 6. Versioning

- `bvl-ds-v1.0` = freeze. Patches (`v1.1`) may add train/val groups and
  fix label errors (with a changelog + new manifest hash); class IDs
  never change within v1.
- Every training run records the exact dataset version + manifest SHA —
  a result without them is not a result.

## 7. Dataset card (required at freeze)

Sources + licenses, per-class instance counts per split, condition quota
measurements, known biases/gaps (e.g. cities covered), QA agreement
numbers, and the explicit list of classes below minimum and what was
done about them.
