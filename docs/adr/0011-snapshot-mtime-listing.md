# ADR-11: Local Parquet snapshots hash the (path, size, mtime) listing

**Status:** accepted

Cheap and stable by default; `--deep-snapshot` switches to content hashing
where mtimes lie (CI caches, fresh checkouts). Golden manifest tests use
deep snapshots for machine-independence. Both modes produce `sha256:` ids;
switching modes flips input hashes once (documented).
