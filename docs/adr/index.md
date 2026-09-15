# Architecture Decision Records

An ADR records one load-bearing decision: the context that forced it, what was decided, and what it costs.
They are the authority on why mbt behaves the way it does.
When something in the code looks wrong - lazy imports everywhere, a gate edit that retrains a model, a champion that passes with a warning - read the ADR before changing it.

An accepted decision is not rewritten to match later thinking.
A decision that changes gets a new ADR, and the old one is marked **superseded** (replaced outright) or **amended** (partly changed), with a link to its successor.
What does get updated in place is a record's account of what has shipped, so that no ADR describes a capability as missing after it lands (ADR-23 carries such an update).

## By subsystem

| Subsystem | Read |
|---|---|
| Adapter boundary and interchange | [1](0001-arrow-interchange.md), [2](0002-local-adapters-in-core.md), [14](0014-plugin-import-hygiene.md), [15](0015-contract-refinements.md), [17](0017-jvm-adapters-and-path-data-access.md) |
| Execution | [3](0003-coordinator-job-split.md) |
| Identity and reproducibility | [4](0004-two-hashes.md), [5](0005-profiles-excluded-from-hashes.md), [12](0012-window-expressions-and-anchor.md), [19](0019-env-freeze-digest-and-manifest-verification.md) |
| Selection, state, and datasets | [7](0007-env-digest-not-modifying.md), [11](0011-snapshot-mtime-listing.md), [13](0013-upstream-datasets-auto-materialize.md), [29](0029-single-relation-datasets.md) |
| Gates and tuning | [6](0006-gate-edits-retrain.md), [8](0008-tuning-never-sees-test.md), [9](0009-champion-reevaluated-in-job.md), [10](0010-missing-vs-unloadable-champion.md), [18](0018-paired-bootstrap-champion-gates.md) |
| Features | [27](0027-declarative-feature-treatment.md) |
| Scoring and monitoring | [20](0020-scoring-resource-and-runtime-champion.md), [21](0021-prediction-store-and-ground-truth-ledger.md), [23](0023-warehouse-batch-scoring.md) |
| Tracking | [28](0028-mlflow-training-only-and-champion-carried-config.md) |
| Task verticals | [24](0024-regression-task-vertical.md) |

## All records

| ADR | Decision | Status |
|---|---|---|
| [1](0001-arrow-interchange.md) | Arrow is the data interchange format | accepted |
| [2](0002-local-adapters-in-core.md) | The local data and compute adapters ship inside mbt-core | accepted |
| [3](0003-coordinator-job-split.md) | Training always runs in a subprocess, apart from the coordinator | accepted |
| [4](0004-two-hashes.md) | Two hashes per node; `state:modified` compares `input_hash` | accepted |
| [5](0005-profiles-excluded-from-hashes.md) | Profiles are excluded from config hashes and stored unrendered | accepted |
| [6](0006-gate-edits-retrain.md) | Gate changes retrain the node | accepted |
| [7](0007-env-digest-not-modifying.md) | `env_digest` changes do not mark nodes modified by default | accepted |
| [8](0008-tuning-never-sees-test.md) | Tuning never sees the test split | accepted |
| [9](0009-champion-reevaluated-in-job.md) | The champion is re-evaluated inside the job, on the challenger's split | accepted |
| [10](0010-missing-vs-unloadable-champion.md) | A missing champion passes with a warning; an unloadable one errors | accepted |
| [11](0011-snapshot-mtime-listing.md) | Local Parquet snapshots hash the (path, size, mtime) listing | accepted |
| [12](0012-window-expressions-and-anchor.md) | Window expressions are hashed; one anchor pins their resolution | accepted |
| [13](0013-upstream-datasets-auto-materialize.md) | Required upstream datasets auto-materialize; selection governs training | accepted |
| [14](0014-plugin-import-hygiene.md) | Plugin modules and parameter models import no ML framework | accepted |
| [15](0015-contract-refinements.md) | Contract refinements beyond the original design sketch | accepted, amended by [29](0029-single-relation-datasets.md) |
| [16](0016-multi-table-inputs-and-key-sampling.md) | Multi-table dataset inputs, key-based sampling, warehouse adapters | superseded by [29](0029-single-relation-datasets.md) |
| [17](0017-jvm-adapters-and-path-data-access.md) | JVM-backed adapters (Spark, H2O) and path data access | accepted |
| [18](0018-paired-bootstrap-champion-gates.md) | Champion gates decide on a paired-bootstrap lower bound | accepted |
| [19](0019-env-freeze-digest-and-manifest-verification.md) | A freeze digest covers the whole environment; `--manifest` verifies it | accepted |
| [20](0020-scoring-resource-and-runtime-champion.md) | Scoring pipelines are a resource kind; champions resolve at run time | accepted, amended by [28](0028-mlflow-training-only-and-champion-carried-config.md) and [29](0029-single-relation-datasets.md) |
| [21](0021-prediction-store-and-ground-truth-ledger.md) | Prediction stores, training-time baselines, and the ground-truth ledger | accepted, amended by [28](0028-mlflow-training-only-and-champion-carried-config.md) |
| [22](0022-population-spine-and-per-table-joins.md) | Population spines, per-table join keys, and label time offsets | superseded by [29](0029-single-relation-datasets.md) |
| [23](0023-warehouse-batch-scoring.md) | Warehouse batch scoring: staged versus native prediction stores | accepted; the staged store ships, the native store awaits live verification |
| [24](0024-regression-task-vertical.md) | Regression is a second task vertical, with name-dispatched metrics | accepted |
| [25](0025-per-table-column-projection.md) | Per-table column projection on multi-table inputs | superseded by [29](0029-single-relation-datasets.md) |
| [26](0026-tracking-experiment-per-node-kind.md) | One tracking experiment per node kind | superseded by [28](0028-mlflow-training-only-and-champion-carried-config.md) |
| [27](0027-declarative-feature-treatment.md) | Declarative feature treatment: transforms, monotone constraints, declared categoricals | accepted |
| [28](0028-mlflow-training-only-and-champion-carried-config.md) | Tracking is training-only, runs are timestamped, and the champion carries its inference config | accepted |
| [29](0029-single-relation-datasets.md) | A dataset reads exactly one relation; the join belongs to dbt | accepted |

## Writing a new ADR

Propose one in the pull request that needs it, as `docs/adr/NNNN-short-title.md` with the next free number, and add it to this index and to the `ADRs` section of `mkdocs.yml`.
Keep the shape the existing records use: a `**Status:**` line, the context, the decisions, and the consequences, including what the decision costs.
If it replaces or changes an earlier record, update that record's status line to link here.
