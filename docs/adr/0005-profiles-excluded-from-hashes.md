# ADR-5: Profiles are excluded from config hashes and stored unrendered

**Status:** accepted

## Decision
Nothing from profiles.yml enters node identity hashes; the manifest stores
the selected target's config *unrendered* (env_var() expressions intact).

## Rationale
Environment must not change node identity: moving dev -> prod must not mark
nodes modified. Unrendered storage keeps secrets out of the manifest (NFR-07).

## Nuance
A `var()` used *inside a spec* whose value differs per target does change
the rendered spec and therefore node identity. That is intentional: such a
var changes training semantics. Environment-only knobs (sample_fraction,
max_tuning_trials) are consumed by adapters/runners from target vars, not
templated into specs, so the golden path is unaffected.
