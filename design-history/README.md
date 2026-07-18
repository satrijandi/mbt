# Design history

These are **pre-implementation planning documents** for mbt, kept for provenance.
They describe the project as it was envisioned before v0.1 shipped and are **not maintained**.
Two files remain, and only because code and docs still cite their identifiers:

- `PRD.md` - product requirements; defines the `FR-*`/`NFR-*` IDs cited in code comments and docs (`FR-RUN-07`, `NFR-08`, ...).
- `TSD.md` - the original technical-design sketch; defines the `TSD §N` section anchors cited throughout the source.

The original vision doc (`PLAN.md`) and task breakdown (`TASK.md`) were removed: they carried no live references and are recoverable from git history.

## Authoritative sources instead

For the current design of the system, read these - they supersede the sketches above:

- **`docs/architecture.md`** - the living map of the `mbt-core` engine (compile pipeline, coordinator/job split, module layout).
- **`docs/adr/`** - Architecture Decision Records are the authoritative design record.
  ADR-15 ("v0 contract refinements beyond the original TSD sketch") explicitly supersedes `TSD.md`.
- **`docs/`** - the published documentation (concepts, spec reference, CLI reference, mlops-alignment, v0.1 status).
- **`CLAUDE.md`** - the working guide and load-bearing decisions.
