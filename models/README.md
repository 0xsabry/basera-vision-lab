# Model registry policy

Large weights are **never** committed to plain git.

- Local working copies live in `models/artifacts/` (gitignored).
- Every artifact that matters gets a registry entry in `models/registry.md`:
  version, source, dataset version, training config, SHA-256, evaluation
  report link, and compatibility verdict.
- For remote storage use GitHub Releases on this private repo, Git LFS, or
  Hugging Face (private) — decided per release; the registry entry is the
  source of truth either way.
- A model is **not** production-eligible until it passes the Phase 4
  evaluation gates and the Phase 5 on-device compatibility test defined in
  `docs/PHASE1_REPORT.md`. Export success alone is not compatibility.
