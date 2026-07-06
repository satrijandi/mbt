# ADR-7: env_digest changes do not mark nodes modified by default

**Status:** accepted

An adapter version bump would otherwise retrain everything. `mbt state diff`
reports the digest delta prominently; `--state-include-env` opts into
treating it as modifying every node. Caveat (documented): schema additions in
new mbt releases can also flip config hashes because specs hash their full
field set; upgrade PRs should expect a one-time full retrain signal.
