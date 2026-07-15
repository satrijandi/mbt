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

Known, documented limitation: redaction is exact-substring based, so a secret transformed by user code (encoded, concatenated, upcased) is no longer recognized.
Do not derive new strings from secrets in hooks.
