# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Please do not open a public issue for security reports.
Use [GitHub private vulnerability reporting](https://github.com/satrijandi/mbt/security/advisories/new), or email satrijandi@gmail.com with the details.
You can expect an acknowledgement within a week.

## What counts

mbt's security model in one paragraph: secrets enter specs and profiles only through `env_var()`; rendered values are tainted and every serialization path (events, run results, manifests, CLI errors, `mbt show`, the generated docs site) redacts them; the manifest stores target config unrendered.
Anything that lets a tainted value reach a stored or displayed surface unredacted is a vulnerability - as is any way for a spec, profile, or hook to execute code outside the documented seams (Python hooks and adapters are code by design; YAML and Jinja specs are not).

## Choosing between `env_var()` and `env()`

`env_var('NAME')` declares the value a secret; `env('NAME')` is the same environment lookup for values that are not.
Use `env()` for schema and database names, hosts, ports, roots, and environment names, and `env_var()` for anything whose disclosure would be an incident.

This is a real choice, not a style preference, because redaction is exact-substring in both directions:

- `env_var()` on a **non**-secret censors that string everywhere it legitimately appears, which corrupts machine-readable output. A value of `1` rewrites `0.1234` to `0.***234` in `run_results.json`.
- `env()` on a secret prints it.

When in doubt use `env_var()`: an over-redacted log is recoverable, a leaked credential is not.

## Known, documented limitations

Redaction is exact-substring based, so a secret transformed by user code (encoded, concatenated, upcased) is no longer recognized.
Do not derive new strings from secrets in hooks.

Very short secrets redact poorly for the same reason - a one- or two-character value matches everywhere.
Prefer whole tokens (a full URI over a bare port) for anything passed through `env_var()`.
