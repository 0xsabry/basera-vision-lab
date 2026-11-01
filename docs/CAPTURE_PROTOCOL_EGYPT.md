# Egyptian Field Capture Protocol — BASERA Vision Lab

Purpose: collect the dataset no public source can give us — pedestrian-POV
images of Egyptian streets and public spaces, with the classes and
conditions BASERA users actually face. **Read fully before capturing.**

## 1. Safety of the capturer (non-negotiable)

- Never walk into traffic while looking at a phone. Use **stop-and-shoot**:
  stop in a safe spot, frame, shoot, move on. For walking-motion sequences,
  work in pairs: one walks with the phone chest-mounted/hanging, the other
  watches the surroundings.
- No capture that endangers the capturer or others to "get the shot" —
  a pothole photo is never worth stepping into a road blindly.

## 2. Legal / sensitivity rules (Egypt)

- **Do not photograph** military or police installations, checkpoints,
  government security buildings, or their personnel/vehicles as subjects.
  If one enters a frame incidentally in a wide street scene, that image is
  deleted at review, not kept.
- No capture inside private property, homes (other than the volunteer's
  own, with written consent), hospitals, or places of worship during
  prayer without explicit permission.
- Public street scenes of everyday life are the target; identifiable
  close-up portraits are **not** — keep faces at scene distance.

## 3. Privacy & consent

- Purpose explained to anyone who asks; a one-page Arabic consent/info
  sheet is carried by every capturer.
- Identifiable subjects photographed deliberately (e.g. a volunteer
  demonstrating stairs) sign/record consent first.
- In stored **work copies**: faces and license plates are blurred.
  Raw originals stay in restricted storage, access-logged, and are never
  shared or published.
- No children as deliberate subjects. No personal data in file names or
  metadata (no names, no exact home addresses; location = district level).

## 4. Capture technique (match the real input distribution)

- Device: ordinary Android phones (the target users' class of device),
  default camera app, no filters, no digital zoom. Record device model in
  the session log.
- Height/angle: handheld at **1.2–1.6 m**, natural walking tilt; include
  imperfect framing — off-center subjects, partial cut-offs, tilted
  horizons. Do not "beautify"; BASERA's real inputs are imperfect.
- Modes per session: stills + short video (1080p30). Video frames are
  extracted later at ≤1 frame/2 s with dedupe (see Dataset Architecture).
- Conditions matrix per route (plan the session to hit quotas):
  repeat key routes at **day / dusk / night**; include indoor low-light
  (building entrances, stairwells, transit stations) where permitted.
- Shoot the **negatives** too: closed manhole covers, ramps (not stairs),
  glass reflections, posters/murals of people and cars, shadows that look
  like holes, parked-cart-like clutter that is not a cart.

## 5. Session discipline (= split groups)

- One session = one route/location block × one time block. Session ID:
  `EG-<city>-<yyyymmdd>-<seq>` (e.g. `EG-CAIRO-20261105-03`).
- Session log (CSV in the dataset manifest): session_id, city, district,
  route description, start/end time, lighting, weather, device,
  capturer id, consent notes, incidents/deletions.
- **All media of a session stays together forever** — it will be assigned
  to exactly one split as a group.

## 6. Priority shot list (per session, aim)

| Subject | Min/session | Notes |
|---|---|---|
| Stairs (public, varied) | 15 | top view (descending POV) is the critical angle + bottom view |
| Potholes / broken pavement | 15 | include subtle ones, not only dramatic holes |
| Poles/bollards in walkway | 10 | head-height signs on poles too |
| Tuk-tuks | 10 | parked + moving (from sidewalk), rear 3/4 view common |
| Vendor carts on sidewalk | 10 | with the remaining passable gap visible |
| Barriers / concrete blocks | 10 | roadworks if present (from safe distance) |
| Doors / entrances | 10 | incl. glass doors (Tier C) with reflections |
| Crosswalks | 5 | worn/faded paint included |
| Dogs (street) | 5 | from normal walking distance, never provoke |
| Vehicles/persons context | 20 | microbus included; night repeats count double value |

## 7. Ingest checklist (same day)

1. Copy media unchanged to `raw/<session_id>/`; compute hashes.
2. Fill the session log; flag deletions (sensitive/privacy) and delete.
3. Blur pass produces `work/` copies (faces/plates).
4. Only `work/` copies go to annotation.
