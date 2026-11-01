# Dataset sources & license verdicts

Checked 2026-11-01. Rule: a source enters a training split only after its
verdict here allows the intended use. "Publicly downloadable" ≠ licensed.
Verdicts distinguish **academic/graduation use** (current project basis)
from **commercial distribution** (future product). This is an engineering
review, not legal advice.

| Source | Content fit for BASERA | License (verified summary) | Academic | Commercial | Notes |
|---|---|---|---|---|---|
| MS COCO 2017 | Tier A/B bootstrap: person, vehicles, dog, furniture, phone… | Annotations CC BY 4.0. **Images are Flickr photos under their own per-image licenses** — the dataset license does not clear the images. | ✅ | ⚠️ per-image risk | Fine-tuning pretrained COCO weights (the current baseline path) is different from redistributing COCO images. Never redistribute COCO images as "our dataset". |
| Open Images V7 (Google) | Door, stairs-adjacent, vehicles, everyday objects; 600 box classes | Annotations CC BY 4.0; images listed under CC BY 2.0 | ✅ | ✅ (with attribution + per-image check) | Best bootstrap source for classes COCO lacks (e.g. Door). Download subsets by class via OID toolkit / FiftyOne. |
| Objects365 | Broad object coverage, incl. some COCO-missing classes | Custom terms; research-oriented, commercial use restricted/unclear | ✅ | ❌ until cleared | Use only if a class cannot be sourced elsewhere; keep out of any commercial-bound run. |
| BDD100K (Berkeley DeepDrive) | Street scenes, day/night, vehicles, traffic lights | Data terms grant educational / research / not-for-profit use | ✅ | ❌ | US dashcam POV (car-mounted), not pedestrian POV — domain gap for a walking blind user. Useful for night/vehicle recall tests, not as primary training POV. |
| Mapillary Vistas | Global street-level segmentation | CC BY-NC-SA (research edition) — **NonCommercial** | ✅ | ❌ | Segmentation masks, not boxes; converting masks→boxes is possible but NC blocks commercial runs. |
| Cityscapes | Urban street segmentation | Custom license, research/non-commercial terms, registration required | ✅ | ❌ | Same mask→box caveat; German cities, low Egypt relevance. |
| Waymo Open Dataset | High-quality driving perception | Non-commercial license | ✅ | ❌ | Vehicle POV + heavy download; low priority for pedestrian use case. |
| VizWiz (Univ. of Colorado) | **Photos actually taken by blind people** — blur, bad framing, occlusion: the real input distribution | Research dataset, request form + terms of use (non-commercial research) | ✅ | ❌ | Not a box dataset for our taxonomy (captions/VQA/saliency/few-shot subsets). Value = **evaluation & domain-gap analysis**: how models fail on blind-taken photos. VizWiz-FewShot has ~10k segmentations / 100 categories / ~4.5k images. |
| ROADWork (CMU) | Road-work zones, barriers, construction | Dataset ODC-By 1.0; embedded BDD/Mapillary subsets keep their own licenses | ✅ | ⚠️ partial | Candidate source for `barrier` class — verify per-subset origin before use. |
| Custom Egyptian capture (own) | **The decisive source**: tuk-tuks, vendor carts, poles, potholes, stairs, microbus, Egyptian streets, day/night | Ours, under our capture protocol (consent/privacy rules in data/README.md) | ✅ | ✅ | Only source that covers Egypt-specific classes and pedestrian POV. Cost is time + annotation, not licensing. |
| Roboflow Universe (community sets) | Pothole/stairs/etc. sets exist | **Per-dataset licenses vary wildly** (CC BY, CC BY-NC, public domain…) | ⚠️ per-set | ⚠️ per-set | Admissible only set-by-set after recording the exact license + author in the dataset card. Default = rejected until checked. |

## Bottom line

- Bootstrap (academic + commercial-clean): **Open Images subsets + COCO
  annotations/pretrained weights + own Egyptian capture**.
- Research/eval only (keep out of commercial-bound runs): BDD100K,
  Mapillary, Cityscapes, Waymo, VizWiz, Objects365.
- The classes that matter most for BASERA safety (`tuk_tuk`, `pothole`,
  `stairs`, `pole`, `vendor_cart`, Egyptian night streets) have **no**
  adequate public source — custom capture is not optional, it is the core
  of Phase 2.
