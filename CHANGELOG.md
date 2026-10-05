# Changelog

All notable changes to the mbt packages, newest first.
Generated from git history by `scripts/generate_changelog.py` - do not edit by
hand; run the script instead (CI checks it with `--check`).

Every release states its **Retraining impact**, because mbt hashes the whole
spec dump: a release that adds a spec field flips every config hash, so the
next `state:modified` build retrains everything (ADR-7). Read that line before
upgrading a project with expensive models.


## v0.2.0 - 2026-10-05

**Retraining impact:** Full retrain, and some specs need editing first. Every model's config hash changes (the spec models gained fields; ADR-7), so the first `state:modified` build after upgrading retrains everything. A dataset that used `inputs:` (multi-table joins, ADR-16/22/25) no longer parses: ADR-29 moved the join upstream, so build the joined relation in your warehouse or dbt and point `source:` at it. Parse now also rejects a train or validation window that overlaps the test window and a YAML file with a duplicate key, and a build rejects a `features.include` entry that matches no column.

- FEEDBACK v6: mbt let overlapping splits, duplicate keys and empty selections through, and gates no test sat on
- A virtualenv release moved the floors job off packaging 23.0
- make lifecycle failed intermittently on Docker and Woodpecker timeouts, and printed no cause
- make lifecycle had no page saying how to reproduce it or which of its output is harmless
- The urllib3 bump left the runner image's extras closure pinning 2.7.0
- Seven urllib3 and virtualenv advisories landed overnight and turned the security job red
- The showcase scheduled retrain, score and monitor DAGs that no make target ever ran Asked to watch the retrain, score and monitor DAGs run, the showcase had no way to do it: `make demo` runs mbt by hand inside the jupyterlab container, and the only thing that ever triggered mbt_retrain, mbt_score or mbt_monitor was test_showcase_scheduling. On a fresh `make up` + `make ci` the DAGs sit in Airflow with no deployable unit pinned (images.env's IMAGE is empty until a push to main bakes one), so even a manual trigger would fail with "images.env has no IMAGE pin yet".
- Ten PyJWT advisories landed overnight and turned the security job red
- The showcase had a reference, a design and a test tier, but no page a user could follow
- A typo in a model's feature list trained it on one feature fewer and every check passed
- The showcase labelled a cohort whose outcome week had not happened yet and held an October population that does not exist
- The showcase modelled a monthly cadence when the story to tell is a weekly model with a 7-day label
- The showcase had grown fourteen lake tables and six targets around a lifecycle that needs one table
- Policy kept living above the seams, so three adapters wrote out the same build and a target var could move a value past every check
- Four review cycles asked whether mbt does the right things, and none asked whether its modules are the right shape
- The cluster build trained a model, then went looking for it in a different H2O cluster
- A training run said nothing about whether the model still held up after its test window
- Every model upload in `make demo` failed because SeaweedFS sized itself by how full the docker disk was
- Following the quickstart word for word failed at `mbt score`, and the docs had drifted in a dozen more places
- `LOAN_APPLY_PROPENSITY_V1_0_0` does not say where the project name ends
- The showcase never read the object store at score time, and called that a Spark limitation
- A pyspark CVE published against the 3.5.2 floor, and only the floors job could see it
- The warehouse plane needed a privilege the operator's role does not have
- Three showcase breaks from ADR-29 that no tier I ran could see
- The floors job caught an arrow call that only exists above the duckdb floor
- The DS primer still taught the join mbt no longer does
- Delete the join: 2,300 lines that three adapters each had to reimplement
- The showcase restated the whole join in its scoring spec, with a comment warning what happens if it drifts
- Moving the join upstream would have made a new feature column look exactly like a data refresh
- The champion recorded exactly which columns it was fit on, and nothing ever read it back
- Counting the actions still on Node 20 by reading runs.using said seven, and the real answer was ten
- The runner was already forcing our pinned actions onto Node 24, and saying so on every run
- The wide probe declared three of its five string columns, and only the nightly live tier could tell
- The CI mbt stamps into every project died at its second step, and no test had ever cloned one
- Retrain a model and MLflow showed two runs with one name, in an experiment called mbt
- Time-anchored features had one answer in mbt, and it was to throw them away
- The floors job found a real advisory, and the fix was already in the lock
- Training and serving runs shared one MLflow experiment, so it filled with operations
- The quickstart's first model failed its own gate, for every new user
- A notebook landed unformatted, and lint-type has gone red on every push since
- The warehouse plane had a make target but no lab bench
- Write down how work lands on main, since the push itself checks nothing
- A padded SNOWFLAKE_SCHEMA seeded twelve tables, then said it does not exist
- Point .gitignore back at the artifacts this repo actually produces
- The upstream tier was reporting my missing tags as an upstream break
- Retire the v3 review to design-history, leaving the repo root to root documents
- Give the changelog guards the git history they read, and stop asserting a file that moves every commit
- Work through the FEEDBACK_v3 review end to end: every finding closed, verified, and documented
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
