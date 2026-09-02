# Changelog

All notable changes to the mbt packages, newest first.
Generated from git history by `scripts/generate_changelog.py` - do not edit by
hand; run the script instead (CI checks it with `--check`).

Every release states its **Retraining impact**, because mbt hashes the whole
spec dump: a release that adds a spec field flips every config hash, so the
next `state:modified` build retrains everything (ADR-7). Read that line before
upgrading a project with expensive models.


## Unreleased

**Retraining impact:** Not yet assessed - assume a full retrain on the next `state:modified` build (ADR-7) until a maintainer records otherwise here.

- Answer two new advisories, and stop matching them by one spelling
- Stop discovering editor checkpoint copies of specs
- Compare the image closure against the committed lock, not the working tree
- Tell somebody when main goes red, too
- Stop resolving typer's control-flow exceptions by module path
- Add issue and PR templates
- Write down the one-time release setup, and check it against the workflows
- ADR-23: record that Spark warehouse scoring landed
- Tell somebody when the nightly goes red
- Ship the PEP 561 markers, so consumers can actually see mbt's types
- Pin the runner image's non-mbt dependency closure
- Make the Spark adapter refuse to guess which address a source is read by
- Create the floors job's venv with --clear, and guard the job itself
- Remove examples/snowflake_wide, porting its unique coverage into the showcase
- Seed only the cadence the Snowflake plane actually reads
- Use the real SeaweedFS credentials on the host-run Snowflake plane
- Load the showcase's Snowflake tables with parquet logical types
- Seed the showcase's Snowflake tables with explicit DDL, not INFER_SCHEMA
- Stop passing empty connect_args through to the Snowflake connector
- Give the showcase a Snowflake data plane (P7, unparked)
- Stop Snowflake SSO opening one browser window per source table
- Lower the h2o floor back under sparkling's backend; guard the pin pair
- Make the floors job actually install floors, and fix what that exposed
- Stop _numeric_ks returning inf; unpin two tests from one dependency version
- Upgrade the locked world; hold pyspark at <4.2, which stopped scoring bit-exactly
- Move the demo projects under tests/fixtures; drop the s3_wide example
- Add the upstream-resolution tier: test the world, not just the lock
- Clear two cryptography advisories; make the third one expire on its own
- Cap h2o below 3.46.0.12, which paywalls MOJO export
- Lock aiohttp and gitpython past this week's advisories
- make workspace: restage over a previous run's root-owned output
- Lock gitpython past the five advisories that turned the security job red
- Control files must be readable by uids other than the writer's
- ADR-25: per-table column projection on multi-table inputs
- snowflake_wide: dev runs must not require prod's key-pair env var
- snowflake_wide: state the column contract per table
- snowflake_wide: document the read-only-sources / writable-sandbox grant layout
- snowflake_wide: the full DS walkthrough - SSO targets, server-side seeding, scoring
- docs: add the DS primer - the training pipeline for data scientists
- Showcase walkthroughs: the DS notebook is the first step after make up
- Showcase: add the DS inner-loop notebook, keep it executable and honest
- Naming conventions: one glossary, uniform inference_date joins across the showcase
- docs/showcase: document the DS / MLOps ownership seam
- README: make the Status section scannable
- SHOW-20: harden the wide cadence into the batch-monthly churn story
- F1 log entry: record the green third release run and the asset cleanup
- Release assets: attach only wheels and sdists (uv build's dist/.gitignore leaked into v0.1.0 as a stray asset, removed by hand)
- Extend the F1 log entry: second release-pipeline bug (publish ordering + opt-in gate) found and fixed by the tag exercise

## v0.1.0 - 2026-07-22

**Retraining impact:** None - first release, so there is no prior manifest to diff against.

- Release: create the GitHub release before the opt-in PyPI publish
- Log the v0.1.0 tag cut and the release-gate envelope fix in the FEEDBACK_v2 progress log
- Release gate: grant the reusable-CI call the callee's full permission envelope
- Work through the FEEDBACK_v2 review end to end: every finding closed, verified, and documented
- Add a live_snowflake test for the wide multi-table example
- Add an S3-lake variant of the wide multi-table example
- Guard the snowflake_wide example with a pytest; share the stub harness
- Document the new event-bus log lines in the troubleshooting runbook
- Add a Snowflake multi-table (wide) example project
- Instrument silent paths with tested event-bus logging
- Add monthly retrain workflow to the scaffold
- Prune superseded design-history sketches (PLAN.md, TASK.md)
- Add contributor-facing architecture doc mapping the mbt-core engine
- Add regression as a second task vertical (XGBoost, LightGBM) - #3, ADR-24
- Share the evaluate -> NodeResult tail between test and evaluate (#4)
- Snowflake batch scoring: build_scoring_input + staged predictions (#1, ADR-23)
- Scaffold installs from tag-pinned git refs; add release workflow (#2)
- Dedup node-lifecycle boilerplate and promote execute-layer seams (#4)
- Harden the test suite: fake compliance, hashing properties, adapter parity
- Add --verbose/-v flag and polish error messages
- Docs accuracy guard + relocate historical planning docs
- Add coordinator error catch-all and de-duplicate check-name registry
- Showcase: demo output pointers cover all three prediction cadences
- Showcase: wide multi-table cadence (SHOW-19) on ADR-22
- ADR-22: population spines, per-table join keys, and label time offsets
- Showcase: share the Airflow task-log volume with the api-server
- Showcase: browser-reachable OAuth login and a browsable lake UI
- Showcase: make clean survives root-owned bind-mount files on native Linux
- Showcase: pin zot to v2.1.16 - v2.1.17+ kills multi-GB blob uploads
- Showcase: standalone-safe modules, exact pins, runbook tier, sharper failure coverage
- Showcase: content-hash image staleness + Woodpecker step-log dumps
- Docs: record the parked P7 Snowflake warehouse variant scope
- Snowflake: add .env.example and document where credentials live
- Snowflake: first-class externalbrowser SSO, checked against connector 4.7.1
- Showcase: add the mbt_score_monthly DAG (SHOW-17's scheduled path)
- Showcase: add the monthly batch churn cadence on the DuckDB plane (SHOW-17)
- Showcase: upgrade the surrounding services to current stable releases
- Docs: correct make down scope and document make clean/score/monitor in the showcase runbook
- Docs: record the coverage-gate lesson and make the verify battery enforce it
- Tests: restore the 100% coverage gate after the review sweep
- Review: whole-repo sweep - correctness, engine guard rails, packaging, CI
- Docs: audit mbt against the ml-ops.org practice catalogue
- Docs: mark design docs historical, drop em dashes, sync roadmap/tutorial details
- Docs: catch README, roadmap, index, and status up with the shipped scope
- Showcase: probe the docker socket GID for airflow's DAG tasks
- Showcase: dockerized full-lifecycle stack, live test tier, docs, nightly CI
- Fix CI: pin test console width, accept unfixable h2o advisory
- Tighten .gitignore: JVM adapter leftovers, local env/secrets guards, editor state
- Scoring + ground-truth monitoring, test sweep, ops docs, team tutorial
- Spark and H2O AutoML adapters: lakehouse data, cluster compute, distributed training
- Snowflake data adapter, multi-table datasets, push-down reproducible sampling
- Hardening: ruff + mypy --strict clean, perf budgets, property tests, status doc
- Compliance suite, LightGBM adapter (G4), churn_demo E2E, ADRs, docs site
- CLI surface, init scaffold, XGBoost/MLflow/Optuna adapters, promote, docs site
- Execution engine: planner, scheduler, runners, training job, gates, state diff
- Compile pipeline: anchoring, snapshot pinning, hashing, deterministic manifest
- Scaffold monorepo; adapter contracts; config, Jinja, parsing, DAG, selectors
