## Repo settings this project assumes

None of these are in any file here, so nothing in this tree can enforce them:

- **The repo is activated in Woodpecker, with these secrets** (repo
  Settings -> Secrets). Woodpecker refuses to start a pipeline whose step
  names a secret that does not exist, so all four must exist:
  - `gitea_token` - a Gitea access token with write access to this repo,
    allowed for the `push`, `pull_request`, `cron` and `manual` events: the PR
    comment, and the `mbt-state` branch the prod build publishes and every
    build reads.
  - `mbt_data_root` - points the prod target at real data, allowed for
    `push`, `cron` and `manual`. `profiles.yml`'s prod target reads
    `{{ env('MBT_DATA_ROOT') }}` with no default, on purpose: the checkout
    only holds the sample data `pr_check.yml` generates, and a prod model
    trained on it would register and score as if it were real. Point
    `sources.yml` at your real tables (relative to that root) at the same time.
  - `mbt_alert_webhook` and `mbt_heartbeat_url` - where failure and breach
    alerts go (Slack, Teams, PagerDuty) and the cron monitor the scheduled
    pipelines ping (healthchecks.io, Cronitor). The value `disabled` turns
    either one off.
- **The schedules are crons in the repo's Woodpecker settings**, not in these
  files: Settings -> Crons, on branch `main`, named exactly as each scheduled
  pipeline's `when: cron:` says - `daily-score` (`0 6 * * *`), `weekly-retrain`
  (`0 5 * * 1`), `weekly-monitor` (`0 7 * * 1`) and, once a model is tagged
  `monthly`, `monthly-retrain` (`0 4 1 * *`). A selector that matches nothing
  fails the run (exit 1) rather than retraining nothing behind a green
  heartbeat, which is why the monthly cron is not created up front. The score
  and monitor pipelines pin their anchor to the cron's hour, so keep the two
  in step if you move one.
- **Timeouts and cancellation are repo settings in Woodpecker**, not pipeline
  keys: set the repo timeout (60 minutes is a sane ceiling for a prod build),
  and leave "cancel previous pipelines" off for push and cron events - an
  overlapping prod build or scheduled run must queue, never be cancelled
  mid-write.
- **`CODEOWNERS` only binds once branch protection requires reviews.**
  Gitea reads it from the repo root and requests reviews from the owners; it
  gates a merge only when `main`'s protection rule requires approvals and
  limits who may approve - and, for an administrator account, when
  "Administrators must follow branch protection rules" is on, or an admin
  merges past the rule without a word.
- **A manual promotion has no approval step.** Woodpecker has no
  equivalent of a GitHub environment's required reviewers: anyone who can run
  a pipeline can run `promote.yml` by hand. The reviewed path is a
  `promotions.yml` change merged under the protection rule above.

Worth adding while you are there: require signed commits on `main`. The
manifest records the commit every model was built from, so an unsigned history
leaves that provenance chain terminating at a SHA nobody attested to.
