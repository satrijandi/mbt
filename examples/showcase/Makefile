# mbt showcase driver. `make up` boots the stack, `make demo` runs the whole
# lifecycle and prints what to look at. See README.md for the golden path.
SHELL := /bin/bash
SHOWCASE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(SHOWCASE_DIR)/../..)
-include $(SHOWCASE_DIR)/.env

export SHOWCASE_WORKSPACE ?= $(HOME)/.cache/mbt-showcase/workspace
export SHOWCASE_RUNNER_IMAGE ?= mbt-showcase-runner:dev
# Overrides read from .env must reach docker compose's ${...:-default}
# interpolation (make does not export include'd variables by itself).
# Undefined names export as empty, which :- treats the same as unset.
export SHOWCASE_S3_PORT SHOWCASE_FILER_PORT SHOWCASE_MLFLOW_PORT SHOWCASE_SPARK_UI_PORT \
  SHOWCASE_JUPYTER_PORT SHOWCASE_PUSHGW_PORT SHOWCASE_PROMETHEUS_PORT SHOWCASE_GRAFANA_PORT \
  SHOWCASE_GITEA_PORT SHOWCASE_WOODPECKER_PORT SHOWCASE_WEBHOOK_PORT SHOWCASE_ZOT_PORT \
  SHOWCASE_AIRFLOW_PORT SHOWCASE_S3_KEY SHOWCASE_S3_SECRET
# Compose project name: override for a parallel stack (the runbook test
# tier boots its own isolated copy this way). Woodpecker step containers
# join the project's default network, hence the derived SHOWCASE_NETWORK.
export SHOWCASE_PROJECT ?= mbt-showcase
export SHOWCASE_NETWORK ?= $(SHOWCASE_PROJECT)_default
# Native Linux: /var/run/docker.sock is root:docker, so airflow-scheduler
# must join that group to run DAG tasks. Docker Desktop resolves the mount
# in its VM where the socket is group 0, and the host stat would be wrong -
# hence the uname guard.
export DOCKER_SOCK_GID ?= $(shell [ "$$(uname -s)" = Linux ] && stat -c %g /var/run/docker.sock 2>/dev/null || echo 0)
COMPOSE := docker compose -p $(SHOWCASE_PROJECT) -f $(SHOWCASE_DIR)/compose/docker-compose.yml \
	--profile core --profile spark --profile dev --profile obs --profile ci --profile orch
EXEC := $(COMPOSE) exec -T --workdir /workspace/project jupyterlab
ANCHOR := 2026-06-30T00:00:00Z
MONITOR_ANCHOR := 2026-07-20T00:00:00Z
GITEA_PORT = $(or $(SHOWCASE_GITEA_PORT),3305)
WOODPECKER_PORT = $(or $(SHOWCASE_WOODPECKER_PORT),8305)
ZOT_PORT = $(or $(SHOWCASE_ZOT_PORT),15000)
AIRFLOW_PORT = $(or $(SHOWCASE_AIRFLOW_PORT),8280)

.PHONY: help image workspace up seed down clean urls demo score monthly wide monitor ci
.DEFAULT_GOAL := help

help:
	@echo "mbt showcase (see README.md for the full runbook):"
	@echo "  make up            build the runner image, boot the stack, seed the lake"
	@echo "  make demo          the whole lifecycle, narrated (build -> promote -> score -> monitor)"
	@echo "  make ci            seed Gitea + Woodpecker (org, repos, OAuth app, activation)"
	@echo "  make monthly       the cluster-free monthly cadence on the DuckDB plane"
	@echo "  make wide          the batch-monthly wide cadence (selection funnel + evidently gates + sparkling AutoML)"
	@echo "  make score         rerun daily scoring (lake sync, mbt score, metric push)"
	@echo "  make monitor       rerun ground-truth monitoring at the pinned anchor"
	@echo "  make inject-drift  poison the scoring batch and watch MbtShiftBreach fire"
	@echo "  make urls          print every UI URL and its login"
	@echo "  make down          stop and remove containers, volumes, and the network"
	@echo "  make clean         down, then also remove the workspace"

image:
	$(SHOWCASE_DIR)/scripts/build_image.sh

# tmp/ and monitoring/ are created by the HOST here on purpose: every container
# runs as root, so whichever one reached the directory first would own it and
# the host user could no longer write into its own workspace.
workspace:
	mkdir -p $(SHOWCASE_WORKSPACE)/tmp $(SHOWCASE_WORKSPACE)/monitoring
	rsync -a --delete $(SHOWCASE_DIR)/project/ $(SHOWCASE_WORKSPACE)/project/
	rsync -a --delete $(REPO_ROOT)/examples/churn_demo/data/ $(SHOWCASE_WORKSPACE)/seed/
	rsync -a $(SHOWCASE_DIR)/data/ $(SHOWCASE_WORKSPACE)/seed/
	rsync -a $(SHOWCASE_DIR)/bootstrap/ $(SHOWCASE_WORKSPACE)/bootstrap/

up: image workspace
	$(COMPOSE) up -d --wait
	$(MAKE) seed urls

seed:
	$(COMPOSE) exec -T jupyterlab python /workspace/bootstrap/seed_lake.py

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
	@echo "  Gitea         http://localhost:$(GITEA_PORT) (after make ci: mbtops / mbtops-showcase-password)"
	@echo "  Woodpecker    http://localhost:$(WOODPECKER_PORT) (after make ci: click login, use the Gitea account)"
	@echo "  Zot           http://localhost:$(ZOT_PORT)/v2/_catalog"
	@echo "  Airflow       http://localhost:$(AIRFLOW_PORT) (admin / admin; DAGs appear after make ci)"
	@echo ""

# The P2 CI loop: seed Gitea (org, churn repo, OAuth app), re-up Woodpecker
# with the real credentials, mint its API token, activate the repo. After
# this, clone http://localhost:$(GITEA_PORT)/mbt-showcase/churn and open a
# PR - Woodpecker runs pr-check and posts the mbt build report comment.
ci:
	@set -euo pipefail; \
	pre=$$(cd $(REPO_ROOT) && uv run python examples/showcase/scripts/ci_bootstrap.py pre \
	  --gitea-url http://localhost:$(GITEA_PORT) --gitea-container $(SHOWCASE_PROJECT)-gitea-1 \
	  --woodpecker-url http://localhost:$(WOODPECKER_PORT) \
	  --project-dir $(SHOWCASE_WORKSPACE)/project \
	  --network $(SHOWCASE_NETWORK) --workspace $(SHOWCASE_WORKSPACE)); \
	jsonval() { printf '%s' "$$pre" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().splitlines()[-1])[sys.argv[1]])" "$$1"; }; \
	export SHOWCASE_WOODPECKER_CLIENT=$$(jsonval client_id); \
	export SHOWCASE_WOODPECKER_SECRET=$$(jsonval client_secret); \
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

# The whole lifecycle, narrated. Anchors are pinned to the seeded data range
# (2026-01..06); MONITOR_ANCHOR sits past the 14d ground-truth maturity.
demo:
	@echo "==> [1/6] DS inner loop: mbt build --target dev (Spark local[2] over the s3a lake, H2O local)"
	$(EXEC) mbt build --target dev --anchor $(ANCHOR)
	@echo "==> [2/6] Cluster training: mbt build --target prod (sparkling H2O inside Spark executors)"
	$(EXEC) mbt build --target prod --anchor $(ANCHOR)
	@echo "==> [3/6] Gate-verified promotion of the freshest staging champions to production"
	$(EXEC) bash -c 'mbt promote --model churn_automl --to production'
	$(EXEC) bash -c 'mbt promote --model churn_monthly_xgb --to production'
	$(EXEC) bash -c 'mbt promote --model churn_wide_automl --to production'
	@echo "==> [4/6] Sync the lake to the scoring plane"
	$(EXEC) python /workspace/bootstrap/sync_lake.py
	@echo "==> [5/6] Batch scoring with the production champion (run-time resolution)"
	$(EXEC) mbt score --target prod_score --anchor $(ANCHOR) --deep-snapshot
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project
	@echo "==> [6/6] Ground-truth monitoring at maturity"
	$(EXEC) mbt monitor --target prod_score --anchor $(MONITOR_ANCHOR) --deep-snapshot
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project
	@echo ""
	@echo "Predictions (one dir per cadence): $(SHOWCASE_WORKSPACE)/lake_local/predictions/"
	@$(MAKE) urls

score:
	$(EXEC) python /workspace/bootstrap/sync_lake.py
	$(EXEC) mbt score --target prod_score --select tag:daily --anchor $(ANCHOR) --deep-snapshot
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project

# The monthly cadence (SHOW-17), entirely on the DuckDB batch plane: retrain
# from the synced lake, gate-verified promote, score the month-start batch.
# `make monitor` evaluates its 30-day labels once they mature.
monthly:
	$(EXEC) python /workspace/bootstrap/sync_lake.py
	$(EXEC) mbt build --target prod_score --select tag:monthly --anchor $(ANCHOR) --deep-snapshot
	$(EXEC) bash -c 'mbt promote --model churn_monthly_xgb --to production'
	$(EXEC) mbt score --target prod_score --select tag:monthly --anchor $(ANCHOR) --deep-snapshot
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project

monitor:
	$(EXEC) mbt monitor --target prod_score --anchor $(MONITOR_ANCHOR) --deep-snapshot
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project

# The wide multi-table batch-monthly cadence (SHOW-19/SHOW-20, ADR-22):
# population spine + three feature histories joined by different keys, the
# ds-helper selection funnel as a committed diff, an Evidently stability
# gate before promotion (train vs test on the selected features; a breach
# exits 2 and stops this recipe before promote), sparkling AutoML on the
# selected columns, the population-form scoring input, and a second
# Evidently gate on the scored batch. `make monitor` evaluates the
# one-month-later outcomes once they mature.
wide:
	$(EXEC) python /workspace/bootstrap/sync_lake.py
	$(EXEC) mbt build --target dev --select churn_wide_probe --anchor $(ANCHOR)
	$(EXEC) python scripts/select_features.py
	$(EXEC) git -C /workspace/project diff --stat -- models/churn_wide_automl.yml || true
	$(EXEC) mbt build --target prod --select churn_wide_automl --anchor $(ANCHOR)
	$(EXEC) python scripts/evidently_gate.py --phase train --export-reference /workspace/monitoring/wide_reference.parquet
	$(EXEC) bash -c 'mbt promote --model churn_wide_automl --to production'
	$(EXEC) mbt score --target prod_score --select tag:wide --anchor $(ANCHOR) --deep-snapshot
	$(EXEC) python scripts/evidently_gate.py --phase serving --reference /workspace/monitoring/wide_reference.parquet
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project
	@echo "drift report: $(SHOWCASE_WORKSPACE)/project/drift_report.html"
	@echo "selection report: $(SHOWCASE_WORKSPACE)/project/target/feature_selection_report.json"

# Break the model in production on purpose, then watch the alert fire:
# shift the scoring batch, score (exit 2 tolerated), push, check Grafana.
inject-drift:
	$(EXEC) python /workspace/bootstrap/inject_drift.py
	-$(EXEC) mbt score --target prod_score --select tag:daily --anchor $(ANCHOR) --deep-snapshot
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project
	@echo "shift injected: watch MbtShiftBreach fire in Prometheus/Grafana; make score to recover"
