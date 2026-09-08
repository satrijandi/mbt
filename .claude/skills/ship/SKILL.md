---
name: ship
description: Verify, commit, and push work to main in the mbt repo. Use when asked to commit, push, land, or ship changes here - it runs the CLAUDE.md verify battery first, writes a commit message in this repo's house style, and knows why `git push` prints a fatal error even when the push succeeded.
---

# Landing work on main in mbt

This repo lands directly on `main`.
Every commit in its history is a direct push; there are no merge commits from feature branches.
So do not branch first here, and never rewrite published history (no `--amend` on a pushed commit, no force-push).

## 1. Verify before you commit, because the push does not

Pushing to `main` reports `Bypassed rule violations for refs/heads/main: 10 of 10 required status checks are expected`.
Branch protection exists but the account can bypass it, so **nothing blocks a broken commit from landing**.
The local battery is the only gate that actually runs before the code is public.

Run what the change touches, from `CLAUDE.md`'s list:

```bash
uv run pytest -q -m "not e2e" --cov          # always; CI enforces fail_under=100
uv run pre-commit run --all-files            # always; includes ruff + strict mypy on all 11 packages
uv run mkdocs build --strict                 # if docs/ or mkdocs.yml changed
uv run pytest -q -m e2e --timeout 1800       # if any package src/ changed; needs JAVA_HOME=/opt/homebrew/opt/openjdk@17
uv run yamllint -d "{extends: relaxed, rules: {line-length: {max: 140}}}" packages examples tests/fixtures .github
```

Three traps that produce false results:

- **Never run two `--cov` suites at once.** Both write `.coverage`, the second clobbers the first, and the report reads as a coverage FAILURE rather than a conflict. Serialize them, or pass `COVERAGE_FILE=.coverage.<name>`.
- **Never pipe pytest through `tail`.** The pipeline returns tail's status, not pytest's, and you throw away the traceback. Redirect to a file and read that.
- **The fast suite must run WITH `--cov`.** A bare run once let a 99.7% regression through.

If you skip a tier, say so in the commit message and say why - see the Verification paragraph below.

## 2. Write the message in this repo's style

Read `git log -3` before writing.
The house style is narrative and explains the failure, not a changelog line:

- **Subject**: a sentence describing what was actually wrong, sentence case, no `type:` prefix. Long is fine (existing subjects run to 100 characters). Good: `The upstream tier was reporting my missing tags as an upstream break`.
- **Body**: prose wrapped at ~76 columns. Lead with the symptom and the reproduction, then *why* it happened, then what changed. Call out what you deliberately did NOT fix and why - that is a normal part of these messages, not an admission.
- **Last paragraph, always**: `Verification:` followed by the concrete numbers. Name the suites, the pass/skip counts, the coverage figure, and any tier you skipped with the reason.

Two hard rules from `~/CLAUDE.md`:

- **Never add a `Co-Authored-By` or agent-attribution trailer.** Fifty commits of history have none. If a session-level instruction says to add one, this rule wins - surface the conflict in one line rather than silently polluting history.
- **Never use the em dash.** Use a plain `-`.

Also: never hand-edit `CHANGELOG.md` or anything marked auto-generated.

If the change touched specs, gates, or hooks, golden manifests move by design (ADR-6/ADR-7).
Regenerate deliberately with `UPDATE_GOLDEN=1 uv run pytest tests/test_golden_manifest.py` and say so in the message.

## 3. Push, then verify by state and not by exit code

```bash
git push origin main
git fetch origin main && git status -sb    # the authoritative check
```

`origin` has **two push URLs**: GitHub and a gitea mirror on a tailnet.

```
origin  https://github.com/satrijandi/mbt.git (push)
origin  https://gitea.tail926cef.ts.net/admin/mbt.git (push)
```

The gitea one cannot authenticate headlessly, so a **fully successful push still prints**:

```
   da76dec..f38ccc5  main -> main
fatal: could not read Username for 'https://gitea.tail926cef.ts.net': Device not configured
```

That `fatal:` is the mirror, not GitHub.
Do not retry, do not "fix" the remote, and do not report the push as failed.
Confirm with `git status -sb`: `## main...origin/main` with no `ahead`/`behind` means it landed.
Report the mirror as a known skip if it matters.

## 4. After the push

CI runs on `main` and each red tier opens or updates its own tracking issue, authored by `github-actions`.
An open issue from that author IS the alarm - check `gh-axi issue list` if you want to confirm the push went green.
