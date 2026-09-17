# ADR-6: Gate changes retrain the node

**Status:** accepted (consequence of ADR-4), amended by [ADR-30](0030-training-report-and-after-test-window.md)

ADR-30 narrows the rule to what gates: after-test gates and
`evaluation.stability` retrain like any gate, while the presentation-only
`evaluation.report` block is excluded from the hash and never does.

Editing a gate threshold changes the rendered spec, flips config_hash, and
marks the model modified. Acceptable v0 cost for hash simplicity; revisit
together with ADR-4's train-only hash if dogfooding shows wasted retrains.
