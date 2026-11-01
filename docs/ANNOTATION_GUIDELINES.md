# Annotation Guidelines — BASERA Vision Lab (Dataset v1)

Tooling: CVAT (self-hosted) or Label Studio; export = YOLO format
(normalized xywh), class IDs from `configs/classes_v2.yaml` — locked.
Annotator language: labels are English class names; this guide is the
single source of truth for edge cases.

## 1. Universal box rules

1. **Tight box** around the visible extent of the object. No padding
   beyond ~2% — loose boxes teach loose localization.
2. Occlusion: annotate if **≥30% of the object is visible** and its
   identity is certain. Box = visible part only; set `occluded=true`.
   Below 30% or uncertain identity → skip the instance (it becomes
   background; uncertain guesses are worse than misses in training data).
3. Truncation: objects cut by the image edge are annotated with the box
   clipped to the image; set `truncated=true`.
4. Minimum size: skip instances whose box would be **<16 px** on the
   shorter side at native resolution (unlearnable noise for a 640 nano).
   Exception: `traffic_light` — annotate down to 10 px (small but vital;
   flagged `difficult=true`).
5. Crowds: individual boxes whenever persons/vehicles are separable;
   a merged "crowd" box is forbidden.
6. Reflections, shadows, posters, screens showing objects: **never**
   annotated as the object. Images containing them as the main trap are
   tagged `hard_negative` at image level.
7. One instance = one box. Never nest duplicates.

## 2. Per-class edge cases (the ones that decide quality)

- **stairs**: box covers the whole flight (all visible steps) as one
  instance. A landing splits two flights into two boxes. Ramps are NOT
  stairs (hard negative). Escalators: annotate as stairs only if steps
  are visible and it functions as stairs in the scene; otherwise skip.
- **pothole**: box = rim of the hole/break in the walking surface.
  Closed manhole covers are NOT pothole (hard negative vs `open_manhole`).
  Cracks without depth: annotate only if a clear surface break ≥ ~15 cm.
- **open_manhole** (Tier C): a cover-less or displaced-cover opening.
  If unsure whether it's open → annotate as pothole, never guess upward.
- **pole**: full visible vertical extent within the walking corridor:
  sign poles, bollards, lamp posts, tree trunks blocking the path get a
  box; poles far off-path (across the road) are skipped — context rule:
  the pole must be on/in the pedestrian space of the scene.
- **barrier**: concrete blocks, metal road barriers, construction
  fencing segments — one box per contiguous barrier run.
- **vendor_cart**: the cart body (with its load silhouette); the vendor
  person beside it is a separate `person` box.
- **tuk_tuk**: three-wheeled Bajaj-type vehicles. A motorcycle with a
  cargo box is still `motorcycle`. Uncertain → skip, flag for review.
- **bus**: includes microbus/minibus. A van without passenger windows
  rows → `truck` if cargo-type, else `car` (review queue decides ties;
  consistency beats individual judgment).
- **door**: the door leaf/frame opening a person would pass through.
  Open doorways without a leaf count. Shopfront full-glass entrances:
  annotate `door` on the passable panel if distinguishable, plus
  `glass_door` (Tier C) on the glass extent — overlap is allowed here.
- **glass_door** (Tier C): transparent/translucent door panels. Box the
  panel extent (handles/frame edges are the cues). Never annotate
  ordinary windows as glass_door.
- **fire** (Tier C): visible flame or dense smoke source region. Steam,
  dust, and fog are NOT fire (hard negatives).
- **crosswalk**: the striped/marked crossing area as one box, even if
  worn. Unmarked informal crossings are NOT crosswalk.
- **dog**: street and pet dogs; statues/toy dogs are hard negatives.
- Tier B merges: **bag** = backpack OR handbag OR suitcase (one class);
  **cup_bottle** = cup OR bottle. Annotate the physical object only,
  not its printed picture on packaging.

## 3. Image-level tags (required)

`lighting` (day/dusk/night/low/backlight) · `blur` (none/mild/strong) ·
`context` (street/indoor/transit) · `hard_negative` (bool + reason) ·
`night` subset membership is derived from lighting.

## 4. Workflow & QA gates

1. Annotator → self-check against §1–2 → submit.
2. **Reviewer pass on 100%** of custom-capture images for Tier A classes
   in v1; 20% sample for public-source subsets (already labeled upstream
   but re-checked for our class merges — merge errors are systematic!).
3. **Double annotation**: 10% of images annotated independently by two
   people. Gate: mean IoU of matched boxes ≥0.85 and class agreement
   ≥95%; below gate → guideline clinic + re-annotation of the batch.
4. A frozen **gold set** (200 images, adjudicated by two reviewers)
   qualifies new annotators (must reach ≥0.90 class agreement on it).
5. Disagreements are adjudicated into this document as new examples —
   the guidelines grow from real disputes, not theory.

## 5. What never enters the dataset

- Images without a recorded source + license (see dataset architecture).
- Images with unblurred deliberate close-up faces in work copies.
- Near-duplicates of an image already in another split (pHash gate).
- AI-generated/upscaled images presented as captures. Synthetic data, if
  ever used, is a separately flagged source with its own ablation —
  never silently mixed into v1.
