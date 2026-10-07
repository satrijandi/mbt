## Repo settings this project assumes

None of these are in any file here, so nothing in this tree can enforce them:

- **The repo variable `MBT_DATA_ROOT` points the prod target at real data.**
  `profiles.yml`'s prod target reads `{{ env('MBT_DATA_ROOT') }}` with no
  default, and the prod build, retrain, scoring and monitor workflows export
  the variable from Settings -> Secrets and variables -> Actions -> Variables.
  Until it is set every prod workflow fails with "environment variable
  'MBT_DATA_ROOT' referenced in profiles.yml is not set" - on purpose: the
  checkout only holds the sample data `pr_check.yml` generates, and a prod
  model trained on it would register and score as if it were real. Point
  `sources.yml` at your real tables (relative to that root) at the same time.
- **The monthly retrain is off until the repo variable `MBT_MONTHLY_RETRAIN`
  is `enabled`.** Tag a model `monthly` first: a selector that matches nothing
  fails the run (exit 1) rather than retraining nothing behind a green
  heartbeat.

- **`CODEOWNERS` only binds once branch protection requires reviews.**
  On its own the file requests reviewers; it does not gate a merge. Turn on
  "Require a pull request before merging" plus "Require review from Code
  Owners" on `main`, or the review gate the file implies does not exist.
- **`promote.yml` runs in the `production` environment**, which is where you add
  required reviewers for a manual promotion dispatch. An environment with no
  reviewers configured approves itself.

Worth adding while you are there: require signed commits on `main`. The
manifest records the commit every model was built from, so an unsigned history
leaves that provenance chain terminating at a SHA nobody attested to.
