# FEEDBACK v4: a practice review of mbt

Review date: 2026-09-09, against `main` (`7758106`).
Scope: the five practice areas named in the brief, read as five separate reviewers over one tree - MLOps, DevSecOps, GitOps, data engineering, data science.
Method: full verification battery run locally first, then a code and docs read, then the two findings that could be settled by experiment were settled that way (a real `git clone` of a scaffolded project driven through the reference CI's own steps, and the sampling digest run through DuckDB directly).
Every finding below carries file evidence.
Findings closed in the three earlier cycles (`design-history/reviews/feedback-v1.md`, `-v2`, `-v3`) are not re-litigated.
This file sat at the repo root as `FEEDBACK.md` while the sweep was in flight; the progress log at the bottom closes it.

## The tree is green

Measured on this machine at review time, not quoted from CI:

| Check | Result |
|---|---|
| `pytest -q -m "not e2e" --cov` | **1440 passed, 66 skipped**, coverage **100.0%** of 10006 statements, 189s |
| `pytest -q -m e2e --timeout 1800` (Java 17) | **79 passed, 6 skipped**, 456s, plus one teardown error explained below |
| `ruff check .` / `ruff format --check .` | clean, 379 files |
| `mypy --strict` (all 11 packages) | clean, 128 source files |
| `yamllint` (packages, examples, fixtures, .github) | clean |
| `mkdocs build --strict` | clean |
| `scripts/audit_dependencies.py` | clean, 6 still-earned acceptances |

The e2e teardown error is mine, not the repo's: the session guard in `conftest.py:99` failed the run because this file appeared in the repo root while the suite was in flight.
That guard is doing exactly what it was written to do, and it is worth recording that it caught a real new root entry within minutes of one existing.

Nothing else in this review is a broken test or a lint failure.
It is entirely about practices that the repo either has not reached yet or believes it has reached and has not.

## The honest top line

This repo's engineering discipline is, measurably, in the top percentile of what a review like this normally finds.
The dependency audit wrapper fails on acceptances that stop firing, so a suppression cannot outlive its reason.
The floors job installs every declared lower bound and then re-asserts that it did.
Three workflows open and close their own tracking issue so a red tier cannot go unread.
`docs/mlops-alignment.md` carries an "Honest gaps" table that names twelve practices the tool does not cover, including ones a vendor would round up.
On the pure data-science axis the work is genuinely strong: paired-bootstrap champion gates, an n-aware KS and chi-square critical value, a Cramer's V leakage screen written in stdlib arithmetic, embargoed temporal windows, and an ADR for each.

The pattern this review keeps finding is narrower, and it is the same pattern the v3 review named a year of commits ago: **the discipline is applied to mbt's own repo and stops at the boundary of what mbt hands the user.**

Three concrete shapes of it:

1. **The reference GitOps pipeline that `mbt init` stamps into every project cannot complete its second step.**
   It has never been run against a real `git clone`, because the one test that covers the loop copies a working directory instead of cloning one (A-1).
2. **The supply-chain posture protects the repo and not the thing the repo produces or ships.**
   Wheels get SLSA provenance from a build that trusts fourteen mutable action refs; artifacts get a recorded SHA-256 that nothing ever compares; the scaffold's promotion workflow splices an untrusted input into a shell line in a way that reaches `--force` (A-2, A-3, B-2).
3. **Three published claims about supply chain and reproducibility are not true of the files they describe**, in a repo whose CLAUDE.md makes "keep its claims exactly true" a rule (B-1).

None of these are architectural.
The corrective work is small, and most of it is in the scaffold rather than in `mbt-core`.

## Priority action items

| # | Finding | Area | Size |
|---|---|---|---|
| 1 | A-1 Reference CI dies at `mbt compile`: `profiles.yml` is generated then gitignored | GitOps | S |
| 2 | A-2 `promote.yml` splices `inputs.version` unquoted, reaching `--force` | DevSecOps | S |
| 3 | A-3 `content_hash` is recorded on every artifact and verified on none | DevSecOps | S |
| 4 | B-1 `requirements.txt` is documented as hash-pinned and is not pinned at all | DevSecOps | M |
| 5 | B-2 Renovate is configured to pin action digests; 0 of 14 are pinned | DevSecOps | S |
| 6 | D-1 `feature_shift` significance has no multiple-comparison control | Data science | M |
| 7 | C-1 `ci.yml` and five scaffold workflows declare no `permissions:` | DevSecOps | S |
| 8 | D-2 S3 artifact upload is single-part and fully buffered | Data engineering | S |
| 9 | D-3 Keyless sampling reshuffles train/test on any schema change | Data engineering | M |
| 10 | C-2, C-3, D-4, E-1 (mode bits, commit signing, batch-percentile caveat, parse banner) | mixed | S |

---

## A. Correctness defects

### A-1. The reference GitOps pipeline cannot run on a fresh checkout

`mbt init` writes `profiles.yml` into the project (`cli/scaffold.py:73-88`), and the scaffold's own `.gitignore` ignores it (`_scaffold/gitignore:12`).
So the file is never committed.
No workflow step creates one on the runner, and `~/.mbt/profiles.yml` does not exist on a GitHub Actions runner.

Reproduced end to end, as a user would hit it: scaffold a project, commit it, `git clone` it the way `actions/checkout` would, and run the reference workflow's own first two steps against a clean `HOME`.

```
$ mbt parse
07:18:47  Parsed 6 resources in 0.02s [OK]          # step 1 passes

$ mbt compile --target dev --deep-snapshot          # step 2 of pr_check.yml
Error: no profiles.yml found
  hint: searched: .../ci_checkout/profiles.yml, .../ci_home/.mbt/profiles.yml.
        Create one (see 'mbt init') or pass --profiles-dir.
EXIT=1
```

This is every scaffolded project's first CI run, and it is red.
It affects `pr_check.yml`, `prod_build.yml`, and all four `scheduled_*.yml`, which is six of the seven shipped workflows.
`promote.yml` is the only survivor, and only because `mbt promote` reads the registry rather than a target.

The reason the suite does not catch it is worth recording, because it is the more general lesson.
`tests/test_cli_basics.py:341` builds its "simulated fresh checkout" with `shutil.copytree(scaffold, checkout, copy_function=shutil.copy)`.
`copytree` copies the working directory, so it carries the untracked, gitignored `profiles.yml` along with it.
The test's own comment explains that `copy_function=copy` was chosen to get fresh mtimes like `actions/checkout`, which is exactly right for the ADR-11 property it was written to prove, and exactly wrong as a stand-in for a clone.

**Fix.** The scaffold's `profiles.yml` contains no literal secrets by construction: the `dev` target has no environment references at all, and `prod` uses `env()` with defaults for all three.
So the simplest correct fix is to stop gitignoring it and keep the `env_var()` discipline as the rule that keeps it safe, which is what `SECURITY.md` already says the model is.
If the project prefers the dbt convention of keeping it out of the repo, then ship `profiles.example.yml` committed and add one `cp` step to each workflow.
Either way, change `test_cli_basics.py` to `git clone` into the fresh checkout rather than `copytree`, which restores what that test was written to assert.

### A-2. `promote.yml` splices an untrusted input into a shell line, and it reaches `--force`

`_scaffold/.github/workflows/promote.yml:44-46`:

```yaml
run: >
  mbt promote --model "${{ inputs.model }}" --to "${{ inputs.to }}"
  ${{ inputs.version && format('--version {0}', inputs.version) || '' }}
```

The third line is unquoted.
A dispatch with `version: 1 --force` renders `mbt promote --model X --to production --version 1 --force`.
`--force` is precisely the documented override for "refusing to promote: gates were not recorded as passed at registration" (`promote.py:86-92`), so an input field that looks like a version number is a gate bypass on the workflow whose entire purpose is enforcing the gate.
The quoted `"${{ inputs.model }}"` is the ordinary quote-break variant of the same problem.

The threat model is bounded, because `workflow_dispatch` needs write access and the job sits behind an `environment: production` approval.
It is still the anti-pattern GitHub's own hardening guide names first, and this is the reference workflow copied into every user project, so it propagates.

**Fix.** Move all three inputs into `env:` and reference them as `"$MODEL"`, `"$TO"`, `"$VERSION"`, with the optional flag chosen in bash rather than in a template expression.

### A-3. Every artifact records a SHA-256 that nothing ever verifies

`content_hash` is computed when an artifact is stored (`storage.py:54` and `:113`), carried into the registry as the `mbt.artifact_content_hash` tag (`mbt_mlflow/adapter.py:297`), and read back into the reconstructed `ArtifactRef` when a champion is resolved (`:314`).
Nothing compares it to the bytes that come back.

Every consumer of `store.fetch()` deserializes immediately:

```
mbt_sklearn/adapter.py:449    payload = joblib.load(store.fetch(ref))
mbt_xgboost/adapter.py:438    booster.load_model(str(store.fetch(ref)))
mbt_h2o/adapter.py:384        shutil.unpack_archive(store.fetch(ref), extract_dir, "zip")
mbt_spark/training.py:355     shutil.unpack_archive(store.fetch(ref), extract_dir, "zip")
mbt_lightgbm/adapter.py:362   payload = json.loads(store.fetch(ref).read_text())
```

The sklearn line is joblib, which is pickle, which is arbitrary code execution, and its own docstring says so (`adapter.py:421`).
The whole point of recording a digest at write time is to be able to detect that the bytes changed between then and now, whether from bucket tampering, a truncated multi-GB download, or a lifecycle rule replacing an object.
The only assertion anywhere is that the string starts with `sha256:` (`compliance/suite.py:353`).

This is the one finding that sits oddly against everything else in the repo.
The release workflow generates signed SLSA provenance for the wheels because "the published wheels were the one gap, which is backwards, they are the only artifacts that leave this repo".
The same argument applies with more force to the model binary, which is the artifact that gets loaded into a production process.

**Fix.** Verify in both stores' `fetch()`: hash the resolved file, compare to `ref.content_hash`, raise an `MbtError` naming both digests on mismatch.
Skip the comparison when the ref carries no hash (older registry entries) rather than failing, and say so in the message.
This is roughly fifteen lines and it closes the loop the write path already opened.

---

## B. Claims that nothing enforces

### B-1. The scaffold's `requirements.txt` is documented as a hash-pinned lock and is neither

`docs/tutorial.md:41` tells a new user the scaffold ships "a hash-pinned `requirements.txt`".
The shipped file is 19 lines, of which three are requirements, and all three are git refs with no hashes and no transitive pins:

```
mbt-core @ git+https://github.com/satrijandi/mbt@v__MBT_VERSION__#subdirectory=packages/mbt-core
mbt-xgboost @ ...
mbt-mlflow @ ...
```

It is byte-for-byte the same set as `requirements.in`, whose header correctly explains how to produce a real lock (`uv pip compile --generate-hashes`) and correctly frames the current file as the state before that.
So the file is honest and the tutorial is not.

Three consequences follow, in ascending order of importance:

- The same sentence says "six GitHub workflows" where `mlops-alignment.md:22` says seven, and seven is right.
  `scheduled_retrain_monthly` is the one missing from the list.
- `requirements.txt`'s own header calls the git ref "an immutable release tag".
  Git tags are movable, on GitHub included; a commit SHA is the immutable form, and the header already offers it as the stopgap.
- The one that matters: **the reference CI's training environment floats.**
  Pinning `mbt-core` pins none of `xgboost`, `mlflow`, `numpy`, `scipy`, or `pyarrow`, and those are the versions that decide model numerics.
  The file's header states the opposite as its reason for existing: "A floating training environment invalidates the manifest's env digest, so CI always installs from this file."
  With three unpinned git refs, `env_freeze_digest` (ADR-19) changes whenever any transitive dependency releases, which is the exact condition ADR-19 exists to detect.
  The tool's headline reproducibility guarantee is therefore off by default in the pipeline the tool generates.

**Fix.** Generate the lock and commit it, which the header already documents how to do; correct the tutorial's two claims; change "immutable release tag" to name the SHA form.
If a full lock is deferred until PyPI publishing is enabled, then say that in the tutorial instead of claiming the property.

### B-2. Renovate is configured to pin action digests, and none of the fourteen actions are pinned

`renovate.json:6` extends `helpers:pinGitHubActionDigests`.
Across the repo's five workflows and the scaffold's seven, all fourteen distinct actions still use mutable refs:

| Ref | Runs with |
|---|---|
| `pypa/gh-action-pypi-publish@release/v1` | `id-token: write` (PyPI Trusted Publishing) |
| `softprops/action-gh-release@v2` | `contents: write` |
| `actions/attest-build-provenance@v2` | `attestations: write`, `id-token: write` |
| `gitleaks/gitleaks-action@v2` | full history |
| `astral-sh/setup-uv@v5`, `actions/checkout@v4`, 8 others | varies |

`release/v1` is a branch, which is the most mutable ref of the set, and it holds the credential that publishes to PyPI.
The `tj-actions/changed-files` compromise is the canonical demonstration that a v-tag can be repointed retroactively.

The gap is specifically that the *intent* is recorded and the *control* does not exist, which is the failure mode `scripts/audit_dependencies.py` was written to prevent one layer up: a stated protection that reads to the next person as "recently reviewed".
Either Renovate is not running on this repo, or its PRs have not been merged, and both are worth knowing.

The scaffold's `renovate.json` omits the preset entirely, so user projects get no path to pinning at all.

**Fix.** Pin all fourteen to SHAs with a `# vX.Y.Z` trailing comment (Renovate maintains both once pinned), add the preset to the scaffold's `renovate.json`, and confirm Renovate is actually enabled on the repo.

---

## C. Security hardening

### C-1. Six workflows declare no `permissions:` block

`ci.yml` is the only one of the repo's five workflows without a top-level `permissions:` (`codeql.yml`, `live.yml`, `release.yml`, and `upstream.yml` all have one).
Seven of its nine jobs therefore inherit whatever the repository default is, which includes the `pip install`, `uv sync`, and third-party action steps.
In the scaffold, `promote.yml` and all four `scheduled_*.yml` also declare none, and `prod_build.yml` grants `contents: write` at workflow scope where only its final publish step needs it.

**Fix.** `permissions: contents: read` at the top of `ci.yml` (the two jobs that need more already declare it locally) and at the top of the five scaffold workflows; in `prod_build.yml`, move `contents: write` down to the job.

### C-2. Control files are written mode 0o666

`artifacts/atomic.py:30` requests `0o666` for the temp file that `os.replace` promotes into `manifest.json` and `run_results.json`.

The comment explaining the choice is right about the bug it fixed: `mkstemp` hardcodes 0600, which made control files private to the writing uid and broke the showcase's container-writes/host-reads path.
But the use case it names is cross-uid **read**, and 0o644 satisfies that.
0o666 additionally grants group and other **write**, which nothing in the rationale asks for.
Under the usual 022 umask the difference is invisible; under `umask 000`, which several common container base images set, `manifest.json` lands world-writable.
That file is integrity-relevant: it carries the `env_digest` and `env_freeze_digest` that ADR-19 verification checks against.

**Fix.** `_CONTROL_FILE_MODE = 0o644`, and keep the comment, adjusting "the permissions an ordinary write would give it" to match.

### C-3. Commits are unsigned, which leaves half the provenance chain open

Branch protection on `main` is real and well configured; I checked it rather than assuming.
All ten CI contexts are required, force pushes and deletions are blocked, and `enforce_admins: false` is the documented deliberate choice the `notify` job's comment explains.

Two settings are off that bear on the provenance story specifically:

- `required_signatures: false`.
  The release workflow generates signed SLSA attestations answering "which workflow, which commit, which runner produced this wheel", and `GitInfo` records the commit into every manifest.
  Both chains terminate at a commit SHA that carries no cryptographic claim about who authored it.
  Signed commits are what make the attested SHA mean something.
- No `required_pull_request_reviews` at all, so a write-capable push to `main` needs no review.
  For a single-maintainer repo that is a defensible call and `CONTRIBUTING.md` now documents the intended flow, but it is worth being explicit that `CODEOWNERS` in the scaffold implies a review gate the reference project will not actually have unless the user turns one on.

**Fix.** Enable `required_signatures` on `main`, and add one line to the scaffold's `README.md` saying that `CODEOWNERS` only binds once branch protection requires reviews.

---

## D. Data science and data engineering

### D-1. `feature_shift` significance has no multiple-comparison control

`quality/monitors.py:_fail_bar` computes a per-feature critical value at the monitor's `significance`, Kolmogorov for numeric and chi-square for categorical.
Each feature is then an independent hypothesis test at that alpha, and each breach is exit 2.

On a 40-feature model with `significance: 0.05` and no real drift, the expected number of features that breach is 2.
So the monitor fires on most clean nightly runs, and the operational response converges on ignoring it, which is worse than the fixed threshold it replaced because a fixed threshold never claimed to be a test.

This is a half-closed finding rather than a new one, which is why it is worth naming.
`feedback-v1.md:73` identified it: "a fixed PSI cutoff across many features generates false breaches from multiple comparisons".
Its **Fix** line listed three items - severity tiers, an n-aware KS critical value, a most-shifted summary - and all three shipped, visibly and well.
The multiple-comparisons half was in the Impact and not in the Fix, so it was never scheduled, and it is not in `mlops-alignment.md`'s "Honest gaps" table either.
Adding `significance` made it sharper, because the whole point of a significance level is a calibrated false-positive rate and this one is not calibrated.

**Fix.** Benjamini-Hochberg across the feature set is the right default for a screening monitor: rank the per-feature p-values, and fail at the largest `k` where `p_(k) <= k/m * alpha`.
Bonferroni (`alpha/m`) is a two-line alternative if the exact p-values are awkward to recover from the critical-value formulation.
Whichever ships, the `significance` docs should say which correction applies and over what family.
If neither ships, this belongs in the "Honest gaps" table, which is where the rest of this project's unclosed methodology lives.

### D-2. S3 artifact upload is single-part and fully buffered

`storage.py:107-109`:

```python
payload = local_path.read_bytes()
key = "/".join(part for part in (self._base, self._prefix, name) if part)
self._client.put_object(Bucket=self._bucket, Key=key, Body=payload)
```

The whole artifact is read into memory, and `put_object` is a single PUT, which S3 caps at 5 GiB.
`fetch()`, the method directly below it, correctly uses `download_file`, which is the managed transfer that does multipart and streams, so the asymmetry looks unintentional rather than chosen.
Spark model directories, H2O MOJO bundles, and large sklearn pipelines are all plausibly in the range where this matters, and the failure lands after the training hours are already spent, which is the case `_s3_client`'s retry config was added to protect against.

`LocalArtifactStore.put_file` has the mild version of the same shape: it `copyfile`s and then `read_bytes()` the copy back purely to hash it, so a 2 GB model is read twice and held once.

**Fix.** `self._client.upload_file(str(local_path), self._bucket, key)` and hash in a streaming loop; use the same streaming hash in the local store.

While in this file, there is no way to request server-side encryption on the put and no `ServerSideEncryption` parameter threaded through.
Bucket-default encryption covers most deployments, so this is a note rather than a finding, but a `sse` key on the artifact-store config would make the control expressible for teams that must demonstrate it per object.

### D-3. Keyless sampling reshuffles train and test on any schema change

`adapters/local/data.py:_digest_columns` falls back to hashing every column when a dataset declares no `sample_key`:

```python
described = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
return [row[0] for row in described]
```

The docstring and ADR-16 both frame this as the slow path.
It is also the unstable path.
Adding a column to a source changes the digest preimage for every row, so hash-bucket membership is recomputed from scratch and rows cross the train/test boundary.

Measured directly against DuckDB with the module's own digest expression, ten rows, one column added, 80/20 boundary:

```
rows: 10   changed train/test side after adding one column: 4  [1, 2, 6, 7]
```

Forty percent of the held-out set turned over because an unrelated column arrived.
The consequences are that metric history in the tracking server stops being comparable across any schema evolution, and that rows previously held out silently enter training.
Within a single run the champion is re-evaluated on the challenger's split (ADR-9), so the promotion decision itself stays fair; it is the longitudinal record that degrades.

The parse-time warning that exists for keyless random splits covers entity straddle ("repeated entities can straddle train and test", `spec-reference.md:294`), which is a different hazard.

Secondary, and unverified: the fallback hashes floating-point columns through `CAST(... AS VARCHAR)`, whose rendering is engine-specific in a way that integer and string key columns are not, so the F19 cross-adapter guarantee likely does not hold on this path either.
Worth a check before deciding the fix.

**Fix.** The cheapest honest option is to extend the existing parse-time warning to say that a keyless dataset's split membership is not stable across schema changes, and to recommend `sample_key`.
The stronger option is to make `sample_key` required for `strategy: random`, since the keyless path has no property that the keyed path lacks.

### D-4. `percentile: batch` makes a feature's value depend on batch composition

ADR-27 chose batch-relative ranking deliberately and argues the case well: a uniform population shift cancels, so the feature is stationary by construction, and no fitted side-car artifact is needed.
Both are true.

The reciprocal property is not recorded anywhere.
Because the rank is computed within whatever batch is being scored (`feature_treatment.py:_percentile_rank`), the same entity with the same raw value receives a different feature value depending on who else is in the batch.
A scoring run over a non-representative slice, say a re-score of only high-tenure customers after a failed batch, re-spreads that slice's `tenure_days` across the full (0, 1] range, so a uniformly high-tenure population presents to the model as if it spanned the whole tenure distribution.
Predictions are also no longer independent across rows, which is a real surprise for anyone reasoning about a batch scorer as a row-wise function.

ADR-27's "Rejected" section considers `percentile: train` and rejects it on plumbing cost, which is a sound call.
It does not state the cost of the option it took.

**Fix.** Documentation, not code.
Add the caveat to ADR-27's Consequences and to the `percentile: batch` row in `spec-reference.md:607`: the transform assumes each scoring batch is a representative sample of the scored population, and a filtered or unusually small batch changes the feature's meaning.
`docs/ds-primer.md:49`, which currently presents it as the strictly stronger lever, should carry the same sentence.

---

## E. Polish

### E-1. The parse banner prints the directory name, not the project name

`cli/main.py:393`:

```python
bus.emit(ParseStarted(project=cli.project_dir.name))
```

Observed in the A-1 reproduction: the same project, cloned into a directory named `ci_checkout`, reports `Parsing project 'ci_checkout'` while `mbt_project.yml` says `name: demo_proj`.
It reads correctly only when the directory happens to match.

Normally cosmetic, except that `profiles.yml` is keyed by the real project name, and the error a user hits when it is wrong is `profiles.yml has no entry for project 'demo_proj'` (`config/profiles.py:173`).
So the banner shows one name and the failure names another, in exactly the situation where someone is reading the banner to find out what name to use.
Any CI checkout directory, any `--project-dir` at a differently named path, and any clone into a renamed folder produces this.

**Fix.** Emit `ParseStarted` after the project config is loaded, or label the field as the directory.

---

## What this review did not find

Recorded so a later cycle does not re-spend the time.

- **Secret handling** is sound end to end.
  Tainting at `env_var()`, exact-substring redaction on every serialization path including `Manifest.to_json`, unrendered target config in the manifest, and the `env()` sibling for non-secrets with `SECURITY.md` explaining why the choice is real rather than stylistic.
  The v3 A-1 finding is fully closed.
- **The manifest and hashing layer** is the strongest part of the codebase.
  Two-hash identity, environment excluded from node identity by ADR-5, `env_digest` plus `env_freeze_digest` verified at `--manifest` execution, and `generated_at == anchor` for byte-identical recompiles.
- **The atomic-write path** now fsyncs data before the rename and the directory after, and uses `O_EXCL` against a planted symlink.
  The `mkstemp` 0600 regression that broke the showcase's container-writes/host-reads path is fixed; only the mode value in C-2 remains.
- **`shutil.unpack_archive(..., "zip")`** in the H2O and Spark adapters is not a zip-slip vector.
  `zipfile.extractall` sanitizes member paths and does not materialize symlinks, so the usual concern does not apply here.
- **The gate engine** has no correctness defects I could find.
  The disparity ratio is direction-agnostic and its one ill-defined input (`r2`) is rejected at parse, the bootstrap falls back to the point estimate and says so, and the no-champion case passes loudly per ADR-10.
- **`docs/mlops-alignment.md`'s "Honest gaps" table** is accurate against the code on every row I spot-checked.
  D-1 and D-4 above are the two candidates for a thirteenth and fourteenth row.

---

## Progress log

**Sweep closed 2026-09-09.** Every finding above is implemented. One entry per item: symptom, fix, verification, docs.

Tree state at close, measured on this machine:

| Check | Result |
|---|---|
| `pytest -q -m "not e2e" --cov` | **1492 passed, 66 skipped**, coverage **100.00%** of 10079 statements, 188s |
| `pytest -q -m e2e --timeout 1800` (Java 17, `--cov-config=tests/coverage-jvm.cfg`) | **80 passed, 6 skipped**, 455s; JVM adapters **84.17%** against the enforced floor of 83 |
| `ruff check .` / `ruff format --check .` | clean, 381 files |
| `mypy --strict` (all 11 packages) | clean, 128 source files |
| `yamllint`, `mkdocs build --strict`, `pre-commit run --all-files` | clean |
| `scripts/audit_dependencies.py` | clean, 6 still-earned acceptances |
| `mbt init` -> `build` -> `promote` -> `score` -> `monitor` -> `docs generate` | all exit 0 on the changed scaffold, from a cold `mbt init` with an empty `HOME` |

One thing that walkthrough is worth recording for, because it looks like a regression and is not. At a *non-default* sample size (`generate_sample_data.py 1200`), `mbt score` exits 2 on a `tenure_days` PSI breach: a 59-row scoring batch against a 402-row baseline is small-sample noise against a fixed 0.25 threshold. It is not caused by D-3's `sample_key`, and measuring both ways shows the key makes it milder rather than worse - keyless, two features breach at 0.36/0.37 instead of one at 0.277. At the documented default (`generate_sample_data.py`, 5000 rows) the whole loop exits 0 with the key and without it.

The two tiers were run **serially** (`CLAUDE.md` records why: both write the same `.coverage` file).

---

### A-1. The reference GitOps pipeline could not run on a fresh checkout

**Symptom.** Reproduced exactly as the review did, and then as a test: scaffold, commit, `git clone` the way `actions/checkout` does, run `pr_check.yml`'s first two steps against an empty `HOME`. `mbt parse` passes; `mbt compile --target dev --deep-snapshot` exits 1 with `no profiles.yml found`, because the scaffold's `.gitignore` hid `profiles.yml` and a runner has no `~/.mbt` to fall back on. Six of the seven shipped workflows died there.

**Fix.** `profiles.yml` is committed. The scaffold's `.gitignore` now carries a NOTE saying it is deliberately not ignored and why; the file's own header says the same at the point of use and repeats that secrets go through `env_var()`, which keeps the value out of the file and redacts it everywhere. `scaffold.py`'s `_install_home_profiles` docstring no longer claims the project copy is gitignored - `~/.mbt` exists so commands run outside the project still find one.

**Verification.** New e2e `test_reference_ci_first_steps_run_on_a_bare_clone` does the whole reproduction: bare origin, commit, `git clone`, empty `HOME` (only `PATH`/`SYSTEMROOT` in the env), `mbt parse`, `mbt compile`. `test_init_scaffold_is_complete_and_parses` now asserts `profiles.yml` is NOT in `.gitignore` and that the file resolves environment values through `env()`/`env_var()` - both halves, since what makes it safe to commit is the discipline and not the absence.

The more general lesson the review drew is fixed too: `test_scaffold_state_branch_loop_end_to_end` built its "simulated fresh checkout" with `shutil.copytree`, which copies the *working directory* - gitignored files included - so the reference pipeline had never once run against the file set a clone actually has. It now pushes to the local bare origin and `git clone`s from it, with `data/` copied in separately (gitignored by design, and copied with fresh mtimes so the ADR-11 property that test exists for still holds).

**Docs.** `docs/tutorial.md` and `docs/quickstart.md` say committed, and why; the scaffold `README.md` layout entry says a gitignored copy leaves every workflow after `mbt parse` failing. `docs/troubleshooting.md` gains `no profiles.yml found on a CI runner but not locally`, with the `profiles.example.yml` + `cp` alternative for organizations that require it out of the repo.

### A-2. `promote.yml` spliced an untrusted input into a shell line, reaching `--force`

**Symptom.** `${{ inputs.version && format('--version {0}', inputs.version) || '' }}` is substituted before bash parses the line, so a dispatch with `version: 1 --force` rendered `mbt promote --model X --to production --version 1 --force`. `--force` is the documented override for "refusing to promote: gates were not recorded as passed at registration" - a gate bypass on the workflow whose purpose is enforcing that gate. `"${{ inputs.model }}"` was the ordinary quote-break variant.

**Fix.** All three inputs move to `env:` and are referenced as `"$MODEL"`, `"$TO"`, `"$VERSION"`, with the optional flag chosen by an `if [ -n "$VERSION" ]` in bash rather than in a template expression. `pr_check.yml` and `prod_build.yml` got the same treatment for their `steps.state.outputs.have_state` splices: those values are workflow-owned and were never exploitable, but this is the reference every user project copies, so the habit is what should propagate.

**Verification.** `tests/test_workflow_supply_chain.py::test_no_template_expressions_inside_run_scripts` parses every workflow in the repo AND the scaffold and fails on any `${{` inside a `run:` script. Zero remain.

**Docs.** The workflow carries the explanation inline, including why quoting is not enough (the quotes are part of the template).

### A-3. Every artifact recorded a SHA-256 that nothing verified

**Symptom.** `content_hash` is computed on store, carried into the registry as `mbt.artifact_content_hash`, and read back into the reconstructed `ArtifactRef` - and never compared to the bytes that come back. Every consumer of `store.fetch()` deserializes immediately; for `mbt-sklearn` that is joblib, which is pickle, which is arbitrary code execution.

**Fix.** `_verify_content_hash(ref, path)` in `mbt/storage.py`, called from both stores' `fetch()`. On mismatch it raises `MbtError` naming both digests and telling the reader not to load it. A ref carrying no digest warns and proceeds rather than failing: baseline and inference-config refs reconstructed from an older champion's registry tags legitimately carry `""` (`runners._baseline_ref`), and refusing to score with such a champion is worse than the check it skips. The S3 store verifies on download only - the cache is the process's own temp dir, so a hit would be re-hashing bytes nobody else could touch.

**Verification.** `test_fetch_rejects_bytes_that_no_longer_match_their_digest` tampers with a stored file and asserts both digests appear in the message; `test_fetch_passes_through_refs_that_carry_no_digest` pins the warn-and-proceed path; `test_s3_fetch_rejects_an_object_that_changed_under_its_key` replaces the object under a key that still resolves, which is the lifecycle-rule case. The check also found a real fixture bug: three tests in `test_inference_config.py` edited a champion's exported config on disk without updating the recorded digest, which is corruption - they now rewrite both through one helper, so they assert on spec divergence rather than on the integrity check.

**Docs.** `SECURITY.md` gains artifact integrity as the second half of the security model; `docs/troubleshooting.md` gains `artifact content hash mismatch` with the verbatim message and the "treat a reproducible mismatch from a clean download as a store-integrity incident" instruction.

### B-1. The scaffold's `requirements.txt` was documented as a hash-pinned lock and was neither

**Symptom.** `docs/tutorial.md` told a new user the scaffold ships "a hash-pinned `requirements.txt`". The file held three git refs, no hashes, no transitive pins - byte-identical to `requirements.in`, whose header correctly framed that as the state *before* a real lock. Two smaller claims were also wrong: "six GitHub workflows" (there are seven; `scheduled_retrain_monthly` was missing from the list) and requirements.txt's own "immutable release tag" (git tags are movable; a commit SHA is the immutable form). The one that mattered: pinning `mbt-core` pins none of xgboost, mlflow, numpy, scipy or pyarrow, so `env_freeze_digest` moved whenever any of them released - the exact condition ADR-19 exists to detect, on by default in the pipeline the tool generates.

**Fix.** Two halves.

*The substantive half*: `mbt init` now stamps exact `==` pins for the numerics stack (numpy, scipy, pandas, pyarrow, scikit-learn, duckdb, xgboost, mlflow) into both requirements files, resolved from the environment that ran `mbt init` via a new `__PINNED_DEPS__` token beside the existing `__MBT_VERSION__` one. Resolved rather than hardcoded so the pins are versions this mbt was actually tested against and cannot rot in the template; a package that is not installed is skipped rather than guessed at.

*The honest half*: a full `--generate-hashes` lock is not possible while mbt installs from a git ref (no wheel hash to record), so the docs now say what the file is - a version-pinned install set whose transitive dependencies still float - and both requirements headers spell out exactly what to run once mbt is on PyPI. "Immutable release tag" is gone; the header names the SHA form and says a tag is movable.

**Verification.** `test_scaffold_pins_the_numerics_stack_it_was_scaffolded_from` asserts each pin matches the running `importlib.metadata.version`; `test_docs_enumerate_the_workflows_that_actually_ship` fails if the tutorial's named list or its count drifts from the shipped set again.

**Docs.** `docs/tutorial.md` (both claims, plus what "version-pinned but not hash-verified" means), scaffold `README.md`, `docs/quickstart.md`, and a `mlops-alignment.md` "Honest gaps" row for the missing hash-verified lock.

### B-2. Renovate was configured to pin action digests; 0 of 14 were pinned

**Symptom.** `renovate.json` has extended `helpers:pinGitHubActionDigests` since it was written, and every action in all twelve workflows still used a mutable ref - including `pypa/gh-action-pypi-publish@release/v1`, a *branch*, holding the PyPI Trusted Publishing credential. The intent was recorded and the control did not exist, which is the failure mode `scripts/audit_dependencies.py` exists to prevent one layer up.

**Fix.** All 51 `uses:` lines across the repo's five workflows and the scaffold's seven are pinned to 40-hex commit SHAs with a trailing `# vX.Y.Z` comment, resolved from each action's current ref through the GitHub API so behavior is unchanged at the moment of pinning. The scaffold's `renovate.json` gains the preset (it had none, so user projects had no path to pinning at all) plus a `description` explaining what a repointed tag costs.

**Verification.** `tests/test_workflow_supply_chain.py::test_actions_are_pinned_to_commit_digests` fails on any ref that is not 40 hex, and separately on a pin with no version comment - a 40-hex string nobody can read is a pin that stops getting reviewed.

**Docs.** `CONTRIBUTING.md`'s provenance paragraph; `docs/v0.1-status.md` NFR-07.

**Not done, and it is a repo setting rather than a file: Renovate has never run here.** There is no dependency-dashboard issue and no `renovate/*` branch on the remote, so the GitHub App is not installed. The preset cannot maintain what it is not running against.

### C-1. Six workflows declared no `permissions:`

**Symptom.** `ci.yml` was the only one of the repo's five without a top-level block, so seven of its nine jobs inherited the repository default - and they are the jobs that run `pip install`, `uv sync`, and third-party actions. In the scaffold, `promote.yml` and all four `scheduled_*.yml` declared none, and `prod_build.yml` granted `contents: write` at workflow scope where only its last two steps need it.

**Fix.** `permissions: contents: read` at the top of `ci.yml` and of the five scaffold workflows; `prod_build.yml` drops to `contents: read` at workflow scope and declares `contents: write` on the one job that pushes the `mbt-state` branch and uploads the manifest.

**Verification.** `test_workflows_declare_least_privilege_permissions` covers all twelve, and requires `contents: read` at workflow scope everywhere except `release.yml`, whose whole purpose is writing.

### C-2. Control files were written mode 0o666

**Symptom.** `atomic.py` requested `0o666` for the temp file `os.replace` promotes into `manifest.json`/`run_results.json`. The comment's rationale is cross-uid **read**, which 0o644 satisfies; 0o666 additionally grants group and other **write**. Invisible under a 022 umask, not under `umask 000`, which several common container base images set - and `manifest.json` carries the `env_digest`/`env_freeze_digest` that ADR-19 verification checks against.

**Fix.** `_CONTROL_FILE_MODE = 0o644`, with the comment extended to say read is the whole requirement and to name the `umask 000` case.

**Verification.** The existing umask parametrization gains `umask 000 -> 0644` (it was `0o002 -> 0o664` before, which the change also corrects to 0644) and every case now additionally asserts no group/other write bit survives.

**Docs.** The existing `Permission denied reading target/manifest.json` runbook entry already described the 0644 behavior and stays correct.

### C-3. Commits are unsigned, which leaves half the provenance chain open

**Symptom.** Verified rather than assumed: `git log --pretty=%G?` reports `N` for every recent commit, no signing key is configured locally, and `required_signatures` is `false` on `main`. The release workflow generates SLSA attestations and `GitInfo` records the commit into every manifest; both chains terminate at a SHA carrying no cryptographic claim about who authored it. `required_pull_request_reviews` is also unset, so the scaffold's `CODEOWNERS` implies a review gate a reference project will not actually have.

**Fix, in this repo.** The scaffold `README.md` gains a "Repo settings this project assumes" section: `CODEOWNERS` only binds once branch protection requires reviews, `promote.yml`'s `production` environment approves itself until reviewers are configured, and signed commits are worth turning on. `CONTRIBUTING.md` records the signing gap explicitly rather than implying it away, and `docs/v0.1-status.md` NFR-07 states it.

**Not done: `required_signatures` is not enabled on `main`.** Enabling it today would reject every push from a machine with no signing key configured, which is this repo's current state - it needs a signing key set up first, and that is the maintainer's decision to make, not a change to land silently mid-sweep.

### D-1. `feature_shift` significance had no multiple-comparison control

**Symptom.** Each feature under a `significance` monitor is an independent hypothesis test at that alpha, and each breach is exit 2. Measured under the null: at 40 features and `significance: 0.05`, **88.5%** of clean runs produce at least one breach. The monitor fires on most clean nightly runs, and the operational response converges on ignoring it - worse than the fixed threshold it replaced, because a fixed threshold never claimed to be a test.

**Fix.** Benjamini-Hochberg across the feature family. `ks_p_value` and `chi2_p_value` were added as the *exact* inverses of the two existing critical-value functions (the KS one deliberately uses the same first-term Kolmogorov approximation `ks_critical_value` inverts), so "p <= alpha" and "D > critical value at alpha" stay the same decision. `benjamini_hochberg_significance` returns the single per-test alpha that reproduces the step-up rule - `k/m * alpha` for the largest rejecting `k`, `alpha/m` when nothing is rejected - which lets every reported `threshold` stay a critical value in the statistic's own units, the thing an operator can read. A family of one returns alpha unchanged, so `prediction_shift` and single-feature models behave exactly as before.

**Verification.** Same measurement after the change: **3.75%** of clean 40-feature runs breach. `test_feature_shift_significance_corrects_for_multiple_comparisons` builds the shape of a clean run (two features just inside a per-feature 0.05, which is the ~2 that 40 tests produce with no drift) and asserts all pass AND that both would have breached uncorrected; `test_multiple_comparison_correction_still_reports_real_drift` asserts one genuinely shifted feature in a 40-feature family still fails, so the correction did not simply blind the monitor; `test_a_family_of_one_is_not_corrected` pins the no-op case; `test_benjamini_hochberg_significance_matches_the_hand_computed_step_up` pins the rule itself.

**Docs.** `docs/spec-reference.md`'s `significance` block names the correction and the family (the model's monitored feature set) and says `prediction_shift` is a family of one. `docs/troubleshooting.md` adds the new informational line to the table of event lines an operator may now see.

### D-2. S3 artifact upload was single-part and fully buffered

**Symptom.** `put_file` read the whole artifact into memory and issued one `put_object`, which S3 caps at 5 GiB - while `fetch()` directly below it already used the managed `download_file`. The asymmetry looked unintentional, and the failure lands after the training hours are spent. `LocalArtifactStore.put_file` had the mild version: `copyfile` then `read_bytes()` the copy back purely to hash it, so a 2 GB model was read twice and held once.

**Fix.** `upload_file` (managed transfer: streams from disk, switches to multipart above its threshold), and a shared `_sha256_file` streaming hash used by both stores' `put_file` and by the A-3 verification path. `size_bytes` comes from `stat()` rather than from a buffer's length.

**Verification.** `test_s3_put_uses_the_managed_multipart_transfer` records the client methods `put_file` actually calls and asserts `upload_file` and not `put_object` - the method identity *is* the property here, since the 5 GiB ceiling is not reachable in a test.

**Not done, deliberately.** The review's server-side-encryption note ("a note rather than a finding") is not implemented: an `sse` key would need per-store configuration threaded through what is currently a bare URI string, which is a design change rather than this finding. Bucket-default encryption covers it for now.

### D-3. Keyless sampling reshuffled train and test on any schema change

**Symptom.** With no `sample_key`, the local adapter hashes every column, so the digest preimage changes when the schema does. Reproduced against DuckDB with the module's own digest expression: ten rows, one column added, an 80/20 boundary - **three of ten rows changed side**; with a `sample_key`, none did. Metric history stops being comparable across any schema evolution, and rows previously held out silently enter training. (Within one run the champion is re-evaluated on the challenger's split per ADR-9, so the promotion decision itself stays fair; it is the longitudinal record that degrades.)

The review's secondary, unverified worry - that `CAST(... AS VARCHAR)` on floats breaks the F19 cross-adapter guarantee on this path - resolves differently than expected, and in a way that mattered for choosing the fix: **there is no cross-adapter guarantee to break, because the other adapters refuse this path entirely.** `mbt-snowflake` and `mbt-spark` both raise on a keyless sample *and* on a keyless random split ("sampling on Snowflake needs a stable row identity", "a random split on Spark needs 'sample_key'"). The all-columns fallback is local-only, so a spec that works locally is a hard error on either warehouse.

**Fix.** Three parts, taking the review's cheap option and its strong option's spirit without a breaking change:
1. The local adapter warns at the exact moment it falls back, naming the column count, the instability, and that the warehouse adapters reject the path outright. It fires for sampling and for random splits, which is broader than parse-time reach: `sample_fraction` comes from target vars, so parse cannot know it.
2. The parse-time keyless-random-split warning now names *both* hazards. It previously covered only entity straddle, which is a different problem.
3. The scaffold's dataset declares `sample_key: user_id`. The reference project's dev target samples at `sample_fraction: 0.5`, so the golden path was itself riding the unstable, non-portable path.

Making `sample_key` mandatory was considered and not taken: the sampling half of the hazard applies to temporal datasets too, so a consistent rule would forbid every keyless dataset in a project that samples at all, and the local adapter is deliberately the low-friction "point at a parquet" path. The warning fires exactly where the hazard is, and the option to harden it to an error remains open.

**Verification.** `test_keyless_split_membership_moves_when_a_column_is_added` builds the dataset, adds an unrelated column to the source, rebuilds, and asserts the keyless test split changed while the keyed one did not - both directions, so it cannot pass by being insensitive. `test_keyless_sampling_and_splitting_warn_that_they_are_unstable` asserts both warnings fire and that neither fires once a key is declared.

**Docs.** `mlops-alignment.md` "Honest gaps" gains a row; `docs/troubleshooting.md` adds the warning to the informational-lines table; the scaffold dataset carries the reason inline.

### D-4. `percentile: batch` makes a feature's value depend on batch composition

**Symptom.** ADR-27 chose batch-relative ranking deliberately and argued it well, but recorded only the property it bought. The reciprocal was nowhere: the rank is computed within whatever batch is being scored, so the same entity with the same raw value gets a different feature value depending on who else is in the batch, and predictions are no longer independent across rows.

**Fix.** Documentation, as the review specified. ADR-27's Consequences gains the cost of the option it took, next to the cost of the one it rejected; `docs/spec-reference.md` turns "two things worth knowing" into three, with the non-representative-batch example (a re-score of only high-tenure customers re-spreads that slice across the full (0, 1] range); `docs/ds-primer.md` carries the same sentence where it currently presents the lever as strictly stronger; `mlops-alignment.md` "Honest gaps" gains the row.

### E-1. The parse banner printed the directory name, not the project name

**Symptom.** `bus.emit(ParseStarted(project=cli.project_dir.name))`. Cloned into `ci_checkout`, a project named `demo_proj` reported `Parsing project 'ci_checkout'`. Cosmetic except that `profiles.yml` is keyed by the real project name, so the very next error reads `profiles.yml has no entry for project 'demo_proj'` - the banner shows one name and the failure names another, in exactly the situation where someone is reading the banner to learn which name to use.

**Fix.** The banner reads the name from `mbt_project.yml` via `load_project`, falling back to the directory name when that file is unreadable - in which case `parse_project` raises the real, fully-diagnosed error on the next line, so the banner steps aside rather than pre-empting it.

**Verification.** `test_parse_emits_parse_started_and_completed` now asserts `Parsing project 'demo'` AND that the pytest tmp directory's name does not appear; `test_parse_banner_falls_back_to_the_directory_when_the_project_is_unreadable` pins the fallback branch and that the real error still surfaces.

---

## Left open, deliberately

Two items in this cycle are repo settings rather than files, and neither can be landed by a commit:

1. **`required_signatures` on `main` (C-3).** Turning it on today rejects every push from this machine: no signing key is configured and no recent commit is signed. It needs a key set up first, so it stays a maintainer decision. `CONTRIBUTING.md`, `docs/v0.1-status.md` and the scaffold README now state the gap rather than implying it away.
2. **Renovate is not installed on this repo (B-2).** No dependency-dashboard issue exists and no `renovate/*` branch has ever been pushed, so the `helpers:pinGitHubActionDigests` preset - in `renovate.json` since it was written, and now in the scaffold's too - has never run. The 51 pins landed by hand in this sweep; keeping them current needs the GitHub App enabled.

One item is a note the review itself declined to raise to a finding, and it stays a note: **server-side encryption on S3 puts (D-2)**. Expressing it per object needs an `sse` key threaded through artifact-store configuration, which is currently a bare URI string - a design change rather than this fix. Bucket-default encryption covers the deployments mbt supports today.
