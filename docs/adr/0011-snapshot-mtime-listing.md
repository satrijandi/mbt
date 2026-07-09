# ADR-11: Local Parquet snapshots hash the (path, size, mtime) listing

**Status:** accepted

Cheap and stable by default; `--deep-snapshot` switches to content hashing
where mtimes lie (CI caches, fresh checkouts). Golden manifest tests use
deep snapshots for machine-independence. Both modes produce `sha256:` ids;
switching modes flips input hashes once (documented).

The scaffolded CI workflows pass `--deep-snapshot` on every compile, build,
and state diff: each CI run is a fresh checkout, so mtime tokens would mark
every dataset modified on every run and the `state:modified` economy loop
would silently degrade to a full retrain. One token scheme per pipeline -
a deep baseline diffed with the default scheme (or vice versa) flags
everything, because the tokens differ by construction.
