# mbt showcase driver. `make up` boots the stack, `make demo` runs the whole
# lifecycle and prints what to look at. See README.md for the golden path.
SHELL := /bin/bash
SHOWCASE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(SHOWCASE_DIR)/../..)
-include $(SHOWCASE_DIR)/.env

export SHOWCASE_WORKSPACE ?= $(HOME)/.cache/mbt-showcase/workspace
export SHOWCASE_RUNNER_IMAGE ?= mbt-showcase-runner:dev
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

.PHONY: image workspace up seed down clean urls demo score monthly monitor ci

image:
	$(SHOWCASE_DIR)/scripts/build_image.sh

workspace:
	mkdir -p $(SHOWCASE_WORKSPACE)/tmp
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
	@echo "  JupyterLab   http://localhost:$${SHOWCASE_JUPYTER_PORT:-8899}"
	@echo "  MLflow       http://localhost:$${SHOWCASE_MLFLOW_PORT:-5501}"
	@echo "  Spark UI     http://localhost:$${SHOWCASE_SPARK_UI_PORT:-8380}"
	@echo "  S3 (lake)    http://localhost:$${SHOWCASE_S3_PORT:-8333}"
	@echo "  Grafana      http://localhost:$${SHOWCASE_GRAFANA_PORT:-3390} (admin/admin)"
	@echo "  Prometheus   http://localhost:$${SHOWCASE_PROMETHEUS_PORT:-9490}"
	@echo "  Pushgateway  http://localhost:$${SHOWCASE_PUSHGW_PORT:-9491}"
	@echo "  Gitea        http://localhost:$(GITEA_PORT) (after make ci: mbtops / mbtops-showcase-password)"
	@echo "  Woodpecker   http://localhost:$(WOODPECKER_PORT) (log in via Gitea OAuth)"
	@echo "  Zot          http://localhost:$(ZOT_PORT)/v2/_catalog"
	@echo "  Airflow      http://localhost:$(AIRFLOW_PORT) (admin/admin; DAGs appear after make ci)"
	@echo ""

# The P2 CI loop: seed Gitea (org, churn repo, OAuth app), re-up Woodpecker
# with the real credentials, mint its API token, activate the repo. After
# this, clone http://localhost:$(GITEA_PORT)/mbt-showcase/churn and open a
# PR - Woodpecker runs pr-check and posts the mbt build report comment.
ci:
	@set -euo pipefail; \
	pre=$$(cd $(REPO_ROOT) && uv run python examples/showcase/scripts/ci_bootstrap.py pre \
	  --gitea-url http://localhost:$(GITEA_PORT) --gitea-container $(SHOWCASE_PROJECT)-gitea-1 \
	  --project-dir $(SHOWCASE_WORKSPACE)/project \
	  --network $(SHOWCASE_NETWORK) --workspace $(SHOWCASE_WORKSPACE)); \
	jsonval() { printf '%s' "$$pre" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().splitlines()[-1])[sys.argv[1]])" "$$1"; }; \
	export SHOWCASE_WOODPECKER_CLIENT=$$(jsonval client_id); \
	export SHOWCASE_WOODPECKER_SECRET=$$(jsonval client_secret); \
	$(COMPOSE) up -d --wait --force-recreate woodpecker-server woodpecker-agent; \
	cd $(REPO_ROOT) && uv run python examples/showcase/scripts/ci_bootstrap.py post \
	  --gitea-url http://localhost:$(GITEA_PORT) --gitea-container $(SHOWCASE_PROJECT)-gitea-1 \
	  --woodpecker-url http://localhost:$(WOODPECKER_PORT) --gitea-token $$(jsonval gitea_token) \
	  --zot-ref localhost:$(ZOT_PORT)/mbt/churn >/dev/null; \
	echo ""; \
	echo "  CI ready:"; \
	echo "  Gitea        http://localhost:$(GITEA_PORT)  (mbtops / mbtops-showcase-password)"; \
	echo "  Woodpecker   http://localhost:$(WOODPECKER_PORT)  (log in via Gitea OAuth)"; \
	echo "  Repo         http://localhost:$(GITEA_PORT)/mbt-showcase/churn"

down:
	$(COMPOSE) down -v --remove-orphans

clean: down
	rm -rf $(SHOWCASE_WORKSPACE)

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
	@echo "==> [4/6] Sync the lake to the scoring plane"
	$(EXEC) python /workspace/bootstrap/sync_lake.py
	@echo "==> [5/6] Batch scoring with the production champion (run-time resolution)"
	$(EXEC) mbt score --target prod_score --anchor $(ANCHOR) --deep-snapshot
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project
	@echo "==> [6/6] Ground-truth monitoring at maturity"
	$(EXEC) mbt monitor --target prod_score --anchor $(MONITOR_ANCHOR) --deep-snapshot
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project
	@echo ""
	@echo "Predictions: $(SHOWCASE_WORKSPACE)/lake_local/predictions/retention_scores/"
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

# Break the model in production on purpose, then watch the alert fire:
# shift the scoring batch, score (exit 2 tolerated), push, check Grafana.
inject-drift:
	$(EXEC) python /workspace/bootstrap/inject_drift.py
	-$(EXEC) mbt score --target prod_score --select tag:daily --anchor $(ANCHOR) --deep-snapshot
	$(EXEC) python /workspace/project/scripts/push_metrics.py /workspace/project
	@echo "shift injected: watch MbtShiftBreach fire in Prometheus/Grafana; make score to recover"
