# ADR-9: Champion evaluation reruns the champion inside the job

**Status:** accepted

Stored champion metrics came from a different data window; a fair
champion/challenger comparison requires identical data and identical metric
code. The coordinator resolves the champion's ArtifactRef pre-submit; the
job loads it and evaluates it on the same pinned test split as the
challenger. Cost: one extra evaluate per gated model.
