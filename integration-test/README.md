# MBT Integration Test

End-to-end integration test that validates the full MBT lifecycle on a Kubernetes cluster. Uses k3d to spin up a local k3s cluster with 12 services, simulating a production-like ML platform environment.

## Prerequisites

- k3d v5.3.0+
- kubectl
- Helm 3
- Docker
- curl, jq
- Python 3.10+ with MBT installed in `.venv` at the repo root

## Infrastructure (12 Services)

`01-setup-infra.sh` creates a k3d cluster (`mbt-integration`: 1 server + 2 agents) and deploys:

| # | Service | Version | Namespace | Deploy Method |
|---|---------|---------|-----------|---------------|
| 1 | PostgreSQL | Bitnami 18.2 | mbt | Helm |
| 2 | SeaweedFS (S3-compatible) | v4.13 | mbt | Helm |
| 3 | Zot Registry | v2.1.14 | mbt | Helm |
| 4 | MLflow | v3.10.0 | mbt | Custom Deployment |
| 5 | Gitea | v1.25.4 | mbt | Helm |
| 6 | Woodpecker CI | v3.13.0 | mbt | Helm (OCI) |
| 7 | Apache Airflow | v3.1.7 | mbt | Helm |
| 8 | H2O Server | 3.46.0.9-1 | mbt | Custom Deployment |
| 9 | JupyterHub | Z2JH | mbt | Helm |
| 10 | Metabase | v0.59 | mbt | Custom Deployment |
| 11 | Prometheus | v3.9.1 | mbt-monitoring | Helm (kube-prometheus-stack) |
| 12 | Grafana | v12.3.2 | mbt-monitoring | Helm (kube-prometheus-stack) |

Three namespaces: `mbt`, `mbt-pipelines` (for Airflow KubernetesExecutor pods), `mbt-monitoring`.

## Service Access

**HTTP services via Traefik Ingress (`*.localhost` on port 80):**

| Service | URL | Auth |
|---------|-----|------|
| MLflow | http://mlflow.localhost | None |
| Airflow | http://airflow.localhost | admin / admin |
| Gitea | http://gitea.localhost | mbt-admin / mbt-admin-password |
| Woodpecker | http://ci.localhost | OAuth via Gitea |
| JupyterHub | http://jupyter.localhost | admin / mbt-jupyter |
| Metabase | http://metabase.localhost | Setup wizard |
| Grafana | http://grafana.localhost | admin / mbt-grafana |

**Non-HTTP services via NodePort:**

| Service | Host Access | Auth |
|---------|-------------|------|
| PostgreSQL | localhost:30432 | admin / admin_password |
| SeaweedFS S3 | localhost:30333 | mbt-access-key / mbt-secret-key |
| H2O | localhost:30054 | None |
| Zot Registry | localhost:30050 | None |
| Gitea SSH | localhost:30022 | SSH key |

## Directory Structure

```
integration-test/
├── 01-setup-infra.sh              # Step 1: Create k3d cluster + deploy 12 services
├── README.md
├── infra/
│   ├── postgres/
│   │   └── init.sql               # 5 databases + seed data (customers, scoring)
│   ├── mlflow/
│   │   └── Dockerfile             # MLflow v3.10.0 custom image
│   ├── airflow/
│   │   └── Dockerfile             # Airflow v3.1.7 custom image
│   ├── mbt-runner/
│   │   └── Dockerfile             # MBT runner image for pipeline pods
│   └── k8s/
│       ├── namespaces.yaml
│       ├── postgres/values.yaml
│       ├── seaweedfs/
│       │   ├── values.yaml
│       │   └── init-buckets-job.yaml
│       ├── zot/values.yaml
│       ├── mlflow/
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   └── ingress.yaml
│       ├── gitea/values.yaml
│       ├── woodpecker/values.yaml
│       ├── airflow/values.yaml
│       ├── h2o/
│       │   ├── deployment.yaml
│       │   └── service.yaml
│       ├── jupyterhub/values.yaml
│       ├── metabase/
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   └── ingress.yaml
│       ├── monitoring/
│       │   └── prometheus-values.yaml
│       └── mbt-pipelines/
│           ├── rbac.yaml
│           └── configmap.yaml
```

## Test Steps (14-Step Workflow)

| Step | Script | Actor | Description | Status |
|------|--------|-------|-------------|--------|
| 1 | `01-setup-infra.sh` | Script | Create k3d cluster and deploy 12 services | Done |
| 2 | `02-init-env.sh` | Script | Initialize environment (seed PG data, configure profiles) | Pending |
| 3 | `03-ds-init-project.sh` | DS | `mbt init ml-pipeline`, create `training-churn-v1.yaml` | Pending |
| 4 | `04-ds-run-training.sh` | DS | Run training pipeline in JupyterHub notebook | Pending |
| 5 | `05-ds-push-to-gitea.sh` | DS | Commit and push to Gitea (new branch) | Pending |
| 6 | `06-ci-woodpecker.sh` | CI | Woodpecker runs: validate, build-image, compile | Pending |
| 7 | `07-de-review-merge.sh` | DE | Review and merge to main in Gitea | Pending |
| 8 | `08-cd-deploy-dag.sh` | CD | On merge: compile + deploy training DAG to Airflow | Pending |
| 9 | `09-verify-airflow.sh` | DS/DE | Verify Airflow run matches notebook result | Pending |
| 10 | `10-ds-view-mlflow.sh` | DS | View results in MLflow | Pending |
| 11 | `11-ds-get-run-id.sh` | DS | Get MLflow run_id | Pending |
| 12 | `12-ds-create-serving.sh` | DS | Create `serving-churn-v1.yaml` with run_id | Pending |
| 13 | `13-ds-serving-pipeline.sh` | DS | Repeat steps 4-7 for serving pipeline (2 DAGs in Airflow) | Pending |
| 14 | `14-verify-predictions.sh` | DS | Verify predictions in S3 + PostgreSQL | Pending |

**Actors:** DS = Data Scientist, DE = Data Engineer, CI = Continuous Integration, CD = Continuous Deployment

## Running

### Step 1: Setup Infrastructure

```bash
cd integration-test
./01-setup-infra.sh
```

The script will:
1. Check prerequisites (k3d, helm, kubectl, docker, curl, jq)
2. Create k3d cluster `mbt-integration` (1 server + 2 agents)
3. Install Traefik ingress controller
4. Deploy PostgreSQL with 5 databases (mlflow_db, airflow_db, warehouse, gitea_db, metabase_db)
5. Deploy SeaweedFS with S3 buckets (mbt-mlflow-artifacts, mbt-pipeline-artifacts)
6. Deploy Zot container registry (NodePort 30050)
7. Build and push custom images (MLflow, Airflow, mbt-runner) to Zot
8. Deploy all application services
9. Configure Gitea-Woodpecker OAuth2 integration
10. Setup RBAC and ConfigMaps for pipeline namespace
11. Run health checks and print access summary

### Teardown

```bash
k3d cluster delete mbt-integration
```

### Check Status

```bash
kubectl get pods -n mbt
kubectl get pods -n mbt-pipelines
kubectl get pods -n mbt-monitoring
```

## Key Integration Points

1. **PostgreSQL**: Single instance with 5 databases. Init via Bitnami `initdb.scriptsConfigMap` using `infra/postgres/init.sql`.
2. **Gitea <-> Woodpecker**: Script creates OAuth2 app in Gitea via API, stores client_id/secret in K8s Secret, Woodpecker references via `extraEnvFrom`.
3. **MLflow <-> SeaweedFS**: MLflow uses `s3://mbt-mlflow-artifacts/` with `MLFLOW_S3_ENDPOINT_URL` pointing to SeaweedFS.
4. **Airflow <-> K8s**: KubernetesExecutor creates pipeline pods in `mbt-pipelines` namespace.
5. **Pipeline pods**: Run in `mbt-pipelines` with ServiceAccount `mbt-runner`, get env vars from ConfigMap `mbt-pipeline-config`.
6. **Zot Registry**: Images at `localhost:30050/mbt/*`, containerd configured to pull from `zot-registry.mbt.svc.cluster.local:5000` inside cluster.

## Troubleshooting

- **k3d cluster won't start:** Ensure ports 80, 443, 30000-30100 are free. Check `docker ps` for conflicting containers.
- **Pods stuck in Init/CrashLoopBackOff:** Check `kubectl logs <pod> -n mbt -c <init-container>` for migration or dependency issues.
- **Airflow migrations:** The chart uses SimpleAuthManager (no FAB DB). If init containers loop, check `alembic_version` in `airflow_db`.
- **H2O not discovering peers:** The headless service needs `publishNotReadyAddresses: true` for DNS-based cluster discovery.
- **Ingress not working:** Verify Traefik is running: `kubectl get pods -n traefik`. Check `*.localhost` resolves to 127.0.0.1.
- **Registry push failures:** Ensure Zot is accessible at `localhost:30050`. Check `curl http://localhost:30050/v2/`.
