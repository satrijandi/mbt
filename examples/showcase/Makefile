# mbt showcase driver. `make up` boots the stack, `make demo` runs the whole
# lifecycle and prints what to look at. See README.md for the golden path.
SHELL := /bin/bash
SHOWCASE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(SHOWCASE_DIR)/../..)
-include $(SHOWCASE_DIR)/.env

export SHOWCASE_WORKSPACE ?= $(HOME)/.cache/mbt-showcase/workspace
# The Woodpecker OAuth app `make ci` created, so every later compose up -
# `make lifecycle` starting Airflow, or a plain `docker compose up` - keeps
# Woodpecker on it. Without this, compose saw the placeholder credentials as
# changed env and recreated Woodpecker unable to log anyone in.
WOODPECKER_ENV = $(SHOWCASE_WORKSPACE)/woodpecker.env
-include $(WOODPECKER_ENV)
export SHOWCASE_WOODPECKER_CLIENT SHOWCASE_WOODPECKER_SECRET
export SHOWCASE_RUNNER_IMAGE ?= mbt-showcase-runner:dev
# Overrides read from .env must reach docker compose's ${...:-default}
# interpolation (make does not export include'd variables by itself).
# Undefined names export as empty, which :- treats the same as unset.
export SHOWCASE_S3_PORT SHOWCASE_FILER_PORT SHOWCASE_MLFLOW_PORT SHOWCASE_SPARK_UI_PORT \
  SHOWCASE_JUPYTER_PORT SHOWCASE_PUSHGW_PORT SHOWCASE_PROMETHEUS_PORT SHOWCASE_GRAFANA_PORT \
  SHOWCASE_ALERTMANAGER_PORT \
  SHOWCASE_GITEA_PORT SHOWCASE_WOODPECKER_PORT SHOWCASE_WEBHOOK_PORT SHOWCASE_ZOT_PORT \
  SHOWCASE_AIRFLOW_PORT SHOWCASE_S3_KEY SHOWCASE_S3_SECRET
# Compose project name: override for a parallel stack (the runbook test
# tier boots its own isolated copy this way). Woodpecker step containers
# join the project's default network, hence the derived SHOWCASE_NETWORK.
export SHOWCASE_PROJECT ?= mbt-showcase
export SHOWCASE_NETWORK ?= $(SHOWCASE_PROJECT)_default
# The docker-socket group airflow-scheduler joins to run DAG tasks, as
# containers see it - which differs per runtime (scripts/docker_sock_gid.sh
# asks the daemon). Probed lazily and at most once per make run, only by
# recipes that run compose, so `make help` never touches docker; set it in
# the environment or .env to skip the probe.
DOCKER_SOCK_GID ?= $(eval DOCKER_SOCK_GID := $(shell $(SHOWCASE_DIR)/scripts/docker_sock_gid.sh))$(DOCKER_SOCK_GID)
COMPOSE_BASE = DOCKER_SOCK_GID=$(DOCKER_SOCK_GID) docker compose -p $(SHOWCASE_PROJECT) \
	-f $(SHOWCASE_DIR)/compose/docker-compose.yml
# Every profile: what exec, down and named-service commands address.
COMPOSE = $(COMPOSE_BASE) --profile core --profile spark --profile dev --profile obs \
	--profile ci --profile orch
# Start just these profiles' services. `make up` boots the data/ML core plus
# observability; the forge (ci) and the scheduler it feeds (orch) start with
# `make ci`, so a DS loop on a small machine never runs them.
compose_up = $(COMPOSE_BASE) $(foreach p,$(1),--profile $(p)) up -d --wait
EXEC = $(COMPOSE) exec -T --workdir /workspace/project jupyterlab
# Anchors come from the DATA, never the clock: the day after the lake's newest
# weekly cohort to build and score, ten days later to monitor
# (project/scripts/lake_anchor.py, the one definition every pipeline, DAG and
# notebook uses). Asked of the lake lazily and at most once per make run, so a
# target that never runs mbt never needs the stack; set either one to override.
ANCHOR ?= $(eval ANCHOR := $(shell $(EXEC) python scripts/lake_anchor.py))$(ANCHOR)
MONITOR_ANCHOR ?= $(eval MONITOR_ANCHOR := $(shell $(EXEC) python scripts/lake_anchor.py --monitor))$(MONITOR_ANCHOR)
GITEA_PORT = $(or $(SHOWCASE_GITEA_PORT),3305)
WOODPECKER_PORT = $(or $(SHOWCASE_WOODPECKER_PORT),8305)
ZOT_PORT = $(or $(SHOWCASE_ZOT_PORT),15000)
AIRFLOW_PORT = $(or $(SHOWCASE_AIRFLOW_PORT),8280)
# Where `make clone` puts the walkthrough's two working copies: outside the
# mbt checkout, so a clone never lands inside the repo that describes it.
SHOWCASE_WORK ?= $(HOME)/showcase-work

.PHONY: help image workspace up seed down clean urls demo lifecycle score monitor outcomes \
  inject-drift reset ci clone protect anchors advance-week week challenger doctor
.DEFAULT_GOAL := help

# The one lake table's size (bootstrap/churn_panel.py): `default` keeps every
# recipe quick; `make seed SCALE=huge` reseeds ~1.6M rows x 320 columns (2.9GB).
SCALE ?= default
PANEL = $(EXEC) python /workspace/bootstrap/churn_panel.py

help:
	@echo "mbt showcase (see README.md for the full runbook):"
	@echo "  make up            build the runner image, boot the stack, seed the lake table"
	@echo "  make seed          (re)write the lake table; SCALE=huge for ~1.6M rows x 320 columns"
	@echo "  make demo          the whole lifecycle, narrated (build -> promote -> score -> monitor)"
	@echo "  make ci            seed Gitea + Woodpecker (org, repos, OAuth app, activation)"
	@echo "  make lifecycle     the same lifecycle as scheduled Airflow DAG runs of the pinned unit"
	@echo "                     (runs make ci first if needed; retrain -> promote -> score -> monitor)"
	@echo "  make clone         clone the churn repo as mbtops (ops/) and mbtds (ds/) into SHOWCASE_WORK"
	@echo "  make protect       CODEOWNERS + branch protection: promotions need mbtops' approval"
	@echo "  make score         score the newest cohort with the production champion, push metrics"
	@echo "  make outcomes      a week passes: the newest cohort's labels land in the table"
	@echo "  make monitor       ground-truth monitoring at the pinned anchor"
	@echo "  make inject-drift  poison the newest cohort and watch MbtShiftBreach fire"
	@echo "  make reset         put the newest cohort back as seeded (undoes outcomes and drift)"
	@echo "  make advance-week  a week passes: label the newest cohort, land the next one"
	@echo "                     (WEEKS=n for several; FUTURE=1 to simulate weeks after today)"
	@echo "  make week          one week of operations: advance, monitor, retrain, promote, score"
	@echo "  make challenger    a weaker challenger meets the champion gate, which refuses it"
	@echo "  make anchors       print the anchors the lake is at"
	@echo "  make doctor        check docker memory, disk, socket group, ports and leftovers"
	@echo "  make urls          print every UI URL and its login"
	@echo "  make down          stop and remove containers, volumes, and the network"
	@echo "  make clean         down, then also remove the workspace"

image:
	$(SHOWCASE_DIR)/scripts/build_image.sh

# tmp/ and monitoring/ are created by the HOST here on purpose: every container
# runs as root, so whichever one reached the directory first would own it and
# the host user could no longer write into its own workspace.
#
# project/ is removed and restaged rather than rsync'd in place. A previous run
# leaves root-owned output inside it - target/, and jupyter's
# .ipynb_checkpoints/ once a DS opens the notebook - which the host cannot
# unlink on native Linux, so `rsync --delete` fails outright instead of
# restaging. Clearing the directory first needs no list of what a container
# might have left, and lands on the same result --delete was aiming at: the
# workspace copy is exactly the committed source. Host first, container as root
# if that fails, the same fallback `clean` uses.
workspace:
	mkdir -p $(SHOWCASE_WORKSPACE)/tmp $(SHOWCASE_WORKSPACE)/monitoring
	rm -rf $(SHOWCASE_WORKSPACE)/project 2>/dev/null || \
	  docker run --rm -v $(SHOWCASE_WORKSPACE):/workspace $(SHOWCASE_RUNNER_IMAGE) \
	    rm -rf /workspace/project
	rsync -a $(SHOWCASE_DIR)/project/ $(SHOWCASE_WORKSPACE)/project/
	rsync -a $(SHOWCASE_DIR)/bootstrap/ $(SHOWCASE_WORKSPACE)/bootstrap/

up: doctor image workspace
	$(call compose_up,core spark dev obs)
	$(MAKE) seed urls

# The lake table is generated, never committed: deterministic at any SCALE,
# written straight into s3://mbt-lake/churn_panel/ (bootstrap/churn_panel.py).
seed:
	$(PANEL) seed --scale $(SCALE)

urls:
	@echo ""
	@echo "  JupyterLab    http://localhost:$${SHOWCASE_JUPYTER_PORT:-8899}"
	@echo "  MLflow        http://localhost:$${SHOWCASE_MLFLOW_PORT:-5501}"
	@echo "  Spark UI      http://localhost:$${SHOWCASE_SPARK_UI_PORT:-8380}"
	@echo "  Lake browser  http://localhost:$${SHOWCASE_FILER_PORT:-8388}/buckets/ (SeaweedFS filer UI)"
	@echo "  Lake S3 API   http://localhost:$${SHOWCASE_S3_PORT:-8333} (signed clients only: mbtadmin/mbtsecret;"
	@echo "                a bare browser GET returns AccessDenied - browse via the filer UI instead)"
	@echo "  Grafana       http://localhost:$${SHOWCASE_GRAFANA_PORT:-3390} (admin / admin)"
	@echo "  Prometheus    http://localhost:$${SHOWCASE_PROMETHEUS_PORT:-9490}"
	@echo "  Pushgateway   http://localhost:$${SHOWCASE_PUSHGW_PORT:-9491}"
	@echo "  Alertmanager  http://localhost:$${SHOWCASE_ALERTMANAGER_PORT:-9493} (delivers to the alert inbox)"
	@echo "  Alert inbox   http://localhost:$${SHOWCASE_WEBHOOK_PORT:-9309}/requests (webhook-sink: CI + model alerts)"
	@echo "  Gitea         http://localhost:$(GITEA_PORT) (started by make ci: mbtops / mbtops-showcase-password)"
	@echo "  Woodpecker    http://localhost:$(WOODPECKER_PORT) (started by make ci: click login, use the Gitea account)"
	@echo "  Zot           http://localhost:$(ZOT_PORT)/v2/_catalog"
	@echo "  Airflow       http://localhost:$(AIRFLOW_PORT) (started by make ci: admin / admin)"
	@echo ""

# The P2 CI loop: seed Gitea (org, churn repo, OAuth app), re-up Woodpecker
# with the real credentials, mint its API token, activate the repo. After
# this, clone http://localhost:$(GITEA_PORT)/mbt-showcase/churn and open a
# PR - Woodpecker runs pr-check and posts the mbt build report comment.
ci:
	$(call compose_up,core obs ci orch)
	@set -euo pipefail; \
	pre=$$(cd $(REPO_ROOT) && uv run python examples/showcase/scripts/ci_bootstrap.py pre \
	  --gitea-url http://localhost:$(GITEA_PORT) --gitea-container $(SHOWCASE_PROJECT)-gitea-1 \
	  --woodpecker-url http://localhost:$(WOODPECKER_PORT) \
	  --project-dir $(SHOWCASE_WORKSPACE)/project \
	  --network $(SHOWCASE_NETWORK) --workspace $(SHOWCASE_WORKSPACE)); \
	jsonval() { printf '%s' "$$pre" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().splitlines()[-1])[sys.argv[1]])" "$$1"; }; \
	export SHOWCASE_WOODPECKER_CLIENT=$$(jsonval client_id); \
	export SHOWCASE_WOODPECKER_SECRET=$$(jsonval client_secret); \
	printf 'SHOWCASE_WOODPECKER_CLIENT=%s\nSHOWCASE_WOODPECKER_SECRET=%s\n' \
	  "$$SHOWCASE_WOODPECKER_CLIENT" "$$SHOWCASE_WOODPECKER_SECRET" > $(WOODPECKER_ENV); \
	$(COMPOSE) up -d --wait --force-recreate woodpecker-server woodpecker-agent; \
	cd $(REPO_ROOT) && uv run python examples/showcase/scripts/ci_bootstrap.py post \
	  --gitea-url http://localhost:$(GITEA_PORT) \
	  --woodpecker-url http://localhost:$(WOODPECKER_PORT) --gitea-token $$(jsonval gitea_token) \
	  --zot-ref localhost:$(ZOT_PORT)/mbt/churn >/dev/null; \
	echo ""; \
	echo "  CI ready:"; \
	echo "  Gitea        http://localhost:$(GITEA_PORT)  (mbtops / mbtops-showcase-password)"; \
	echo "  Woodpecker   http://localhost:$(WOODPECKER_PORT)  (click login, use the Gitea account)"; \
	echo "  Repo         http://localhost:$(GITEA_PORT)/mbt-showcase/churn"

# The walkthrough's two working copies (docs/showcase-walkthrough.md step 3):
# ops/ as the platform owner, ds/ as the data scientist. Credentials ride in
# the remote URL - throwaway demo logins, so this is the lab's only honest
# option - and each clone commits under its own persona. An existing clone is
# left alone.
clone:
	@set -euo pipefail; mkdir -p $(SHOWCASE_WORK); \
	for spec in ops:mbtops:mbtops-showcase-password ds:mbtds:mbtds-showcase-password; do \
	  IFS=: read -r dir user pass <<< "$$spec"; dest=$(SHOWCASE_WORK)/$$dir; \
	  if [ -d "$$dest/.git" ]; then echo "  $$dest already cloned; skipping"; continue; fi; \
	  git clone -q "http://$$user:$$pass@localhost:$(GITEA_PORT)/mbt-showcase/churn.git" "$$dest"; \
	  git -C "$$dest" config user.name "$$user"; \
	  git -C "$$dest" config user.email "$$user@example.com"; \
	  echo "  $$dest  (as $$user)"; \
	done

# Put promotions under review (walkthrough step 8): CODEOWNERS gives
# promotions.yml to mbtops, and main's protection rule lets only mbtops push
# it, requires one approval from mbtops per merge, and blocks the Gitea
# administrator override. Safe to re-run.
protect:
	@cd $(REPO_ROOT) && uv run python examples/showcase/scripts/ci_bootstrap.py protect \
	  --gitea-url http://localhost:$(GITEA_PORT)

down:
	$(COMPOSE) down -v --remove-orphans

# Containers write into the bind-mounted workspace as root on native Linux
# (Docker Desktop maps writes to the host user), so a plain rm can hit
# Permission denied - fall back to wiping the contents through a container.
clean: down
	rm -rf $(SHOWCASE_WORKSPACE) 2>/dev/null || { \
	  docker run --rm -v $(SHOWCASE_WORKSPACE):/workspace $(SHOWCASE_RUNNER_IMAGE) \
	    bash -c 'rm -rf /workspace/* /workspace/.[!.]* 2>/dev/null || true'; \
	  rm -rf $(SHOWCASE_WORKSPACE); \
	}

# The whole lifecycle, narrated, on the one lake table. ANCHOR is the day
# after the lake's newest weekly cohort lands (2026-09-29 as seeded), which
# makes that cohort the scoring batch; MONITOR_ANCHOR sits past the 7d
# ground-truth maturity.
demo:
	@echo "==> [1/8] DS inner loop: mbt build --target dev (Spark local[2] over the s3a lake, H2O local)"
	$(EXEC) mbt build --target dev --anchor $(ANCHOR)
	@echo "==> [2/8] Cluster training: mbt build --target prod (sparkling H2O inside Spark executors)"
	$(EXEC) mbt build --target prod --anchor $(ANCHOR)
	@echo "==> [3/8] Gate-verified promotion of the freshest staging champion to production"
	$(EXEC) mbt promote --model churn_automl --to production
	@echo "==> [4/8] A weaker challenger: the champion gate refuses it, production stays put"
	$(EXEC) python /workspace/bootstrap/weaker_challenger.py $(ANCHOR)
	@echo "==> [5/8] Batch scoring of the newest cohort with the production champion"
	$(EXEC) mbt score --target batch --anchor $(ANCHOR)
	$(EXEC) python scripts/push_metrics.py /workspace/project
	@echo "==> [6/8] Monitoring before the outcomes land: the labels are still NULL, so it waits"
	$(EXEC) mbt monitor --target batch --anchor $(MONITOR_ANCHOR)
	@echo "==> [7/8] The outcome week closes: the newest cohort's labels land in the same table"
	$(PANEL) land-outcomes
	@echo "==> [8/8] Ground-truth monitoring: realized metrics and gates"
	$(EXEC) mbt monitor --target batch --anchor $(MONITOR_ANCHOR)
	$(EXEC) python scripts/push_metrics.py /workspace/project
	@echo ""
	@echo "Predictions: $(SHOWCASE_WORKSPACE)/predictions/retention_scores/"
	@$(MAKE) urls

# The same lifecycle the way production runs it: nothing by hand in Jupyter.
# Woodpecker's prod-build bakes the deployable unit and pins its digest in the
# deploy repo, git-sync hands the deploy repo's DAGs to Airflow, and every
# retrain, score and monitor below is an Airflow DAG run executing mbt inside
# that pinned unit (scripts/lifecycle.py triggers each run, waits for it, and
# prints its task log and Airflow URL). Promotion goes the reviewed way: a
# promotions.yml PR from the data scientist, the code owner's approval under
# `make protect`, and the promote pipeline the merge runs. It stays a registry
# event, never a deploy (ADR-20): the score run after it serves the new
# champion from the same unit. `make ci` runs first when the forge has not
# been seeded yet.
LIFECYCLE := cd $(REPO_ROOT) && uv run python examples/showcase/scripts/lifecycle.py
AIRFLOW_URL = http://localhost:$(AIRFLOW_PORT)

lifecycle:
	@curl -fsS -o /dev/null http://localhost:$(GITEA_PORT)/api/v1/repos/mbt-showcase/churn 2>/dev/null \
	  || $(MAKE) ci
	$(call compose_up,core spark obs ci orch)
	@echo "==> [1/8] Deployable unit: prod-build baked and pinned by digest, DAGs live in Airflow"
	@$(LIFECYCLE) unit --gitea-url http://localhost:$(GITEA_PORT) \
	  --woodpecker-url http://localhost:$(WOODPECKER_PORT) --airflow-url $(AIRFLOW_URL)
	@echo "==> [2/8] The newest cohort as seeded: its outcome week is still open, labels NULL"
	$(PANEL) reset
	@echo "==> [3/8] Airflow mbt_retrain: mbt build --target prod in the unit (cluster + sparkling H2O)"
	@$(LIFECYCLE) run mbt_retrain --airflow-url $(AIRFLOW_URL)
	@echo "==> [4/8] Governed promotion: mbtds opens a promotions.yml PR, mbtops approves, CI promotes"
	@$(MAKE) --no-print-directory protect >/dev/null
	@$(LIFECYCLE) promote --gitea-url http://localhost:$(GITEA_PORT) \
	  --woodpecker-url http://localhost:$(WOODPECKER_PORT) \
	  --mlflow-url http://localhost:$${SHOWCASE_MLFLOW_PORT:-5501}
	@echo "==> [5/8] Airflow mbt_score: the newest cohort, scored by the run-time production champion"
	@$(LIFECYCLE) run mbt_score --airflow-url $(AIRFLOW_URL)
	@echo "==> [6/8] Airflow mbt_monitor before the outcomes land: nothing matured, so it waits"
	@$(LIFECYCLE) run mbt_monitor --airflow-url $(AIRFLOW_URL)
	@echo "==> [7/8] A week passes: the newest cohort's outcomes land in the same table"
	$(PANEL) land-outcomes
	@echo "==> [8/8] Airflow mbt_monitor: realized metrics and gates against the landed labels"
	@$(LIFECYCLE) run mbt_monitor --airflow-url $(AIRFLOW_URL)
	@echo ""
	@echo "  Airflow DAG runs  $(AIRFLOW_URL)/dags  (admin / admin)"
	@echo "  Predictions       $(SHOWCASE_WORKSPACE)/predictions/retention_scores/"
	@echo "  Grafana           http://localhost:$${SHOWCASE_GRAFANA_PORT:-3390}  (mbt Model Health)"

score:
	$(EXEC) mbt score --target batch --anchor $(ANCHOR)
	$(EXEC) python scripts/push_metrics.py /workspace/project

outcomes:
	$(PANEL) land-outcomes

monitor:
	$(EXEC) mbt monitor --target batch --anchor $(MONITOR_ANCHOR)
	$(EXEC) python scripts/push_metrics.py /workspace/project

# Break the model in production on purpose, then watch the alert fire:
# shift the newest cohort, score (exit 2 tolerated), push, check Grafana.
inject-drift:
	$(PANEL) inject-drift
	-$(EXEC) mbt score --target batch --anchor $(ANCHOR)
	$(EXEC) python scripts/push_metrics.py /workspace/project
	@echo "shift injected: MbtShiftBreach fires in Prometheus and lands in the alert inbox (localhost:$${SHOWCASE_WEBHOOK_PORT:-9309}/requests); make reset score to recover"

reset:
	$(PANEL) reset

# Break a model on purpose and watch the champion gate keep it out of
# production: a working copy drops the activity features and retrains on the
# shared registry (bootstrap/weaker_challenger.py). Needs a production champion
# to compare against, so run it after make demo.
challenger:
	$(EXEC) python /workspace/bootstrap/weaker_challenger.py $(ANCHOR)

anchors:
	@echo "build/score anchor  $(ANCHOR)"
	@echo "monitor anchor      $(MONITOR_ANCHOR)"

# A week passes in the lake: the newest cohort's outcomes land and the next
# Monday's cohort arrives, still open. Every anchor follows, because they all
# derive from the newest cohort. The generator refuses to pass today's date
# unless FUTURE=1 says to simulate weeks that have not happened yet.
WEEKS ?= 1
advance-week:
	$(PANEL) advance --weeks $(WEEKS) $(if $(FUTURE),--simulate-future)
	@$(MAKE) --no-print-directory anchors

# One week of operations after the lake moves on, in the order a team runs
# them: evaluate last week's predictions against the outcomes that just
# landed, retrain on the newer month, let the gates decide whether the
# retrained model may replace production, and score the new cohort. Run it a
# few times (FUTURE=1 past today) and Grafana's panels become time series.
# The promotion step may refuse - that is the champion gate doing its job,
# and the scoring step then serves the champion it kept.
week:
	@echo "==> [1/5] A week passes: last cohort labelled, the next one lands"
	$(PANEL) advance $(if $(FUTURE),--simulate-future)
	@echo "==> [2/5] Ground-truth monitoring of last week's predictions"
	$(EXEC) mbt monitor --target batch --anchor $$($(EXEC) python scripts/lake_anchor.py --monitor)
	$(EXEC) python scripts/push_metrics.py /workspace/project
	@echo "==> [3/5] Weekly retrain on the newer month (champion gate vs production)"
	@set -u; anchor=$$($(EXEC) python scripts/lake_anchor.py); \
	rc=0; $(EXEC) mbt build --target dev --anchor "$$anchor" || rc=$$?; \
	case $$rc in \
	  0) echo "==> [4/5] The retrain passed its gates and was staged: promote it"; \
	     $(EXEC) mbt promote --model churn_automl --to production ;; \
	  2) echo "==> [4/5] The champion gate refused this week's retrain (exit 2): production keeps its champion" ;; \
	  *) exit $$rc ;; \
	esac
	@echo "==> [5/5] Score the new cohort with whatever holds production"
	$(EXEC) mbt score --target batch --anchor $$($(EXEC) python scripts/lake_anchor.py)
	$(EXEC) python scripts/push_metrics.py /workspace/project

doctor:
	@python3 $(SHOWCASE_DIR)/scripts/doctor.py $(if $(FIX),--fix)
