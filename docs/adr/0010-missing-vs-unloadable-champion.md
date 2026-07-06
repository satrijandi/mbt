# ADR-10: Missing champion passes with WARN; unloadable champion errors

**Status:** accepted

Bootstrap must not block the first model: `compare_to` gates pass with an
explicit warning and `champion_version: null` when no champion exists.
A champion that exists but cannot be loaded (missing artifact, format
mismatch) is a hard error - silent skips would rubber-stamp promotions.
