# ADR-12: Window expressions are hashed; one manifest-wide anchor pins resolution

**Status:** accepted

Windows are stored and hashed as expressions ("-28d:now"); one UTC anchor,
taken from the clock at compile time (or --anchor), resolves them into
concrete ranges stored outside the hashed config. Reruns via --manifest
reproduce exactly (G2) while mere time passage never marks nodes modified;
new data arrives as a snapshot change, which does. `generated_at` equals the
anchor by design so same-anchor compiles are byte-identical (FR-COMP-04).
