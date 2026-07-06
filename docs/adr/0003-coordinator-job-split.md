# ADR-3: Coordinator/job split; training always runs in a subprocess

**Status:** accepted

## Decision
The coordinator plans, schedules, resolves champions, evaluates gates, and
registers. Everything needing an ML framework or training data runs in
`python -m mbt.execute.job <job.json>` driven by a ComputeAdapter.

## Rationale
Real `--threads` parallelism (the GIL never serializes training), crash and
memory isolation, pure gate logic in core ("adapters compute metrics, core
compares them"), and the exact serialization seam K8s/Ray reuse in v1
(FR-V1-01). The JobResult returns through a result file; stdout carries the
forwarded event stream.
