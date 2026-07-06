# ADR-6: Gate changes retrain the node

**Status:** accepted (consequence of ADR-4)

Editing a gate threshold changes the rendered spec, flips config_hash, and
marks the model modified. Acceptable v0 cost for hash simplicity; revisit
together with ADR-4's train-only hash if dogfooding shows wasted retrains.
