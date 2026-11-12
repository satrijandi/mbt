# ADR-13: Required upstream datasets auto-materialize; selection governs training

**Status:** accepted; amended by [32](0032-target-path-and-exact-node-execution.md), which adds `--no-auto-materialize` for orchestrators that run one node per task

A model-only PR on a cold CI runner must not fail or force selecting the
world: every dataset a selected model needs joins the execution plan
(cache-aware). Dataset builds are cheap; training is the 1000x cost.
