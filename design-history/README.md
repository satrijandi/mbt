# Design history

Provenance, not documentation.
Nothing in this directory is maintained, and nothing in it should be treated as a description of how mbt works today.

## Pre-implementation planning documents

These describe the project as it was envisioned before v0.1 shipped.
Two files remain, and only because code and docs still cite their identifiers:

- `PRD.md` - product requirements; defines the `FR-*`/`NFR-*` IDs cited in code comments and docs (`FR-RUN-07`, `NFR-08`, ...).
- `TSD.md` - the original technical-design sketch; defines the `TSD §N` section anchors cited throughout the source.

The original vision doc (`PLAN.md`) and task breakdown (`TASK.md`) were removed: they carried no live references and are recoverable from git history.

## Closed review cycles

`reviews/` holds whole-repo reviews whose findings have been worked through and closed.
They carry the reasoning behind a large share of the current design, and several code comments cite them by section (`FEEDBACK 2.6`, `FEEDBACK 3.6`, `R2-7`, `F17`, ...), which is why they are kept rather than deleted.

- `reviews/feedback-v1.md` - reviews 1 and 2, closed 2026-07-20. Section IDs `2.6`, `3.6`, ... and `R2-*`.
- `reviews/feedback-v2.md` - review 3, closed 2026-07-22. Finding IDs `F1`-`F27`, `P3`.
- `reviews/feedback-v3.md` - review 4, closed 2026-09-01. Finding IDs `A-1`-`G-2`, cited as `FEEDBACK v3 A-1`.

There is no review in flight right now.
While one is, it lives at the repo root as `FEEDBACK_v<n>.md` so it is impossible to miss, and moves here once its progress log is closed.

## Authoritative sources instead

For the current design of the system, read these - they supersede the sketches above:

- **`docs/architecture.md`** - the living map of the `mbt-core` engine (compile pipeline, coordinator/job split, module layout).
- **`docs/adr/`** - Architecture Decision Records are the authoritative design record.
  ADR-15 ("v0 contract refinements beyond the original TSD sketch") explicitly supersedes `TSD.md`.
- **`docs/`** - the published documentation (concepts, spec reference, CLI reference, mlops-alignment, v0.1 status).
- **`CLAUDE.md`** - the working guide and load-bearing decisions.
