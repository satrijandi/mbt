# Design history

These are the **pre-implementation planning documents** for mbt, kept for provenance.
They describe the project as it was envisioned before v0.1 shipped and are **not maintained**.

- `PLAN.md` - vision, rationale, roadmap (the original *why*).
- `PRD.md` - product requirements (`FR-*`/`NFR-*` IDs still referenced by the codebase).
- `TSD.md` - the original technical-design sketch (the *how*).
- `TASK.md` - the original task breakdown and sequencing.

## Authoritative sources instead

For the current design of the system, read these - they supersede the sketches above:

- **`docs/adr/`** - Architecture Decision Records are the authoritative design record.
  ADR-15 ("v0 contract refinements beyond the original TSD sketch") explicitly supersedes `TSD.md`.
- **`docs/`** - the published documentation (concepts, spec reference, CLI reference, mlops-alignment, v0.1 status).
- **`CLAUDE.md`** - the working guide and load-bearing decisions.

Requirement IDs (`FR-RUN-07`, `NFR-08`, ...) in code comments and docs still point at `PRD.md`,
so the file is retained rather than deleted.
