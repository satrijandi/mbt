# ADR-19: Freeze digest for the full environment; --manifest verifies it

**Status:** accepted

`env_digest` is a targeted signal: Python, mbt packages, and the packages adapters fingerprint.
A numpy or scipy bump can change model numerics without touching any of those, so the manifest also records `env_freeze_digest`: a sha256 over every installed distribution (pip-freeze-like, sorted `name==version`), computed identically at compile time and at verification time.

Executing a stored manifest is the reproducibility mechanism, so it now verifies the environment instead of silently proceeding.
An `env_digest` mismatch is a hard error; `--allow-env-mismatch` downgrades it to a warning for deliberate cross-environment runs.
A freeze-digest-only mismatch warns: the fingerprinted packages match, but transitive dependencies drifted.
Manifests from schema N-1 have no freeze digest and skip that check.

ADR-7 is unchanged: `state:modified` keys its opt-in env signal to the targeted `env_digest`; the freeze digest is reported in `mbt state diff` for visibility, never as a retrain trigger, because any dev-tool bump would otherwise retrain everything under `--state-include-env`.
