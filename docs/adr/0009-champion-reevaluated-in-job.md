# ADR-9: Champion evaluation reruns the champion inside the job

**Status:** accepted

Stored champion metrics came from a different data window; a fair
champion/challenger comparison requires identical data and identical metric
code. The coordinator resolves the champion's ArtifactRef pre-submit; the
job loads it and evaluates it on the same pinned test split as the
challenger. Cost: one extra evaluate per gated model.

**Shipped behavior, corrected 2026-10-07.**
"Identical data" means the same test ROWS, not the challenger's view of them.
The job scored the champion through the challenger's transformed dataset, so
a champion whose features the challenger had dropped was scored without
those columns - H2O reads a missing column as all-NA - and a clearly worse
challenger passed a paired-bootstrap gate against the crippled champion (the
showcase's weaker-challenger step found it: PR-AUC 0.20 against production's
0.33, with a reported delta lower bound of +0.015). The job now scores the
champion through its own registered spec and fitted feature columns (the
ADR-28 inference config, as scoring and the pre-deploy check already did),
over the same base test split, and refuses a pairing whose row counts
differ. A champion registered before ADR-28 has no such config and is still
scored through the challenger's view.
