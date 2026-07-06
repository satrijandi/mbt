# ADR-8: Tuning never sees the test split

**Status:** accepted

Trials train on the train split and evaluate on validation: the dataset's
declared validation split, else a deterministic implicit carve (temporal:
last 20% by time; random: seeded 20% with seed+2). Implicit carves are
reabsorbed by the final fit; an explicitly declared validation split stays
held out because the user said so. Gates stay honest.
