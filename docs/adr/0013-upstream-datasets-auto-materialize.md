# ADR-13: Required upstream datasets auto-materialize; selection governs training

**Status:** accepted

A model-only PR on a cold CI runner must not fail or force selecting the
world: every dataset a selected model needs joins the execution plan
(cache-aware). Dataset builds are cheap; training is the 1000x cost.
