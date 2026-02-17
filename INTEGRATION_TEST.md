# MBT Integration Test Specification

End-to-end integration test validating the full MBT workflow: from `mbt init` through Airflow-scheduled training and serving pipelines, using a self-hosted infrastructure stack.

## End Goals

1. **Trained model viewable in MLflow** -- metrics, parameters, artifacts all browsable in MLflow UI
2. **Prediction scores stored in PostgreSQL** -- `churn_predictions` table with scored results
3. **Airflow DAGs visible and triggerable** -- scheduled training (monthly) and serving (daily) DAGs in Airflow UI

## Personas

| Persona | Owns | Tools |
|---------|------|-------|
| **Data Scientist (DS)** | `pipelines/*.yaml`, `lib/` custom transforms | `mbt` CLI, MLflow UI, Gitea |
| **Data Engineer (DE)** | `profiles.yaml`, `.woodpecker.yml`, `Dockerfile`, infrastructure | Gitea, Woodpecker, Airflow, K8s |

---

## 1. Architecture Overview

```
                          +-----------+
                          |   Gitea   |
                          | ds-team   |
                          | de-team   |
                          +-----+-----+
                                | webhook
                                v
                          +-----------+
                          | Woodpecker|
                          |    CI     |
                          +-----+-----+
                                |
                   +------------+------------+
                   |                         |
                   v                         v
            +------+------+          +-------+------+
            |     Zot     |          |   Airflow    |
            |  Registry   |          | (DAG deploy) |
            | (mbt-runner)|          +-------+------+
            +------+------+                  |
                   |                         v
                   |                 +-------+----------+
                   +--------------->| K8s Pods          |
                                    | (pod-per-step)    |
                                    +---+-----+-----+--+
                                        |     |     |
                          +-------------+     |     +------------+
                          v                   v                  v
                   +------+-----+     +-------+------+    +------+-----+
                   | PostgreSQL |     |  SeaweedFS   |    | H2O Server |
                   |            |     | (S3-compat)  |    |  (AutoML)  |
                   | - warehouse|     |              |    +------------+
                   | - mlflow_db|     | Bucket 1:    |
                   +------+-----+     |  mlflow-     |
                          |           |  artifacts   |
                          |           |              |
                          |           | Bucket 2:    |
                          |           |  pipeline-   |
                          |           |  artifacts   |
                          |           +-------+------+
                          |                   |
                          |           +-------+------+
                          +---------->|    MLflow    |
                                      | (tracking)  |
                                      +-------------+
```

### Data Flow: Training Pipeline

```
PostgreSQL(customers) -[K8s pod]-> load_data
    -> split_data -> train_model -[H2O Server]-> evaluate -> log_run -[MLflow]-> done
```

Each `->` crosses a K8s pod boundary. Artifacts pass through SeaweedFS Bucket 2 between pods. `log_run` writes model artifacts to SeaweedFS Bucket 1 (via MLflow) and metrics to PostgreSQL `mlflow_db`.

### Data Flow: Serving Pipeline

```
MLflow(SeaweedFS Bucket 1) -[K8s pod]-> load_model ----+
PostgreSQL(customers_to_score) -[K8s pod]-> load_scoring_data --+--> apply_transforms
    -> predict -> publish -[K8s pod]-> PostgreSQL(churn_predictions)
```

---

## 2. MBT Features Required (To Build)

Everything below must be implemented before this integration test can run. Ordered by dependency.

### Priority 1: Core Infrastructure Adapters

| Feature | Type | Package / File | Description |
|---------|------|----------------|-------------|
| `mbt-postgres` | DataConnectorPlugin | `mbt-postgres/` (new) | Read/write PostgreSQL tables. `read_table()` -> `pd.read_sql`, `write_table()` -> `df.to_sql()`. Config: host, port, database, schema, user, password. |
| `mbt-s3` | StoragePlugin | `mbt-s3/` (new) | S3-compatible artifact storage (SeaweedFS). `put/get/exists/list_artifacts`. Uses `boto3` with custom `endpoint_url`. URI: `s3://bucket/run_id/step/artifact`. |
| `mbt step execute` | CLI command | `mbt-core/src/mbt/cli.py` | Single-step execution for K8s pods. Loads manifest, reads inputs from S3, executes one step, writes outputs to S3. |

### Priority 2: Orchestrator & DAG Generation

| Feature | Type | Package / File | Description |
|---------|------|----------------|-------------|
| `mbt-airflow` | OrchestratorPlugin | `mbt-airflow/` (new) | Generates Airflow DAG Python file from manifest. Maps each step to `KubernetesPodOperator` calling `mbt step execute`. |
| `mbt generate-dags` | CLI command | `mbt-core/src/mbt/cli.py` | Reads compiled manifests from `target/*/manifest.json`, generates DAG files via orchestrator plugin. |

### Priority 3: Framework & Runner

| Feature | Type | Package / File | Description |
|---------|------|----------------|-------------|
| Remote H2O support | Adapter extension | `mbt-h2o/src/mbt_h2o/framework.py` | Extend `setup()` to accept `h2o_url` from profile config. `h2o.init(url=...)` instead of local cluster. |
| MBT Runner Dockerfile | Container image | `Dockerfile` (new) | Python 3.10 + all MBT packages + JRE (H2O) + libpq-dev (PostgreSQL). Pushed to Zot. |

### Priority 4: Refactor Hardcoded Steps

These existing steps hardcode `LocalFileConnector` and `LocalStoragePlugin`. They must read the configured plugin from profile config instead.

| File | Current | Required |
|------|---------|----------|
| `mbt-core/src/mbt/steps/load_data.py` | Hardcodes `LocalFileConnector()` | Use `PluginRegistry.get("data_connectors", config_type)` |
| `mbt-core/src/mbt/steps/load_scoring_data.py` | Hardcodes `Path("sample_data")` | Same as above |
| `mbt-core/src/mbt/steps/publish.py` | Only writes CSV | Support `destination: database` via `DataConnectorPlugin.write_table()` |
| `mbt-core/src/mbt/core/runner.py` | Hardcodes `LocalStoragePlugin()` | Use `PluginRegistry.get("storage", config_type)` from manifest |

---

## 3. Infrastructure Setup

All services run via Docker Compose for local testing. For production-like testing, equivalent K8s manifests should be used.

### 3.1 PostgreSQL

```yaml
postgres:
  image: postgres:16
  ports:
    - "5432:5432"
  environment:
    POSTGRES_USER: admin
    POSTGRES_PASSWORD: admin_password
  volumes:
    - ./infra/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    - pg_data:/var/lib/postgresql/data
```

**Two databases:**
- `mlflow_db` -- MLflow backend store (experiment metadata, run metrics, parameters, tags)
- `warehouse` -- Feature/label tables and prediction output

**Tables in `warehouse`:**

```sql
-- Feature + label table (seeded with months of historical data)
CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    tenure INT,
    monthly_charges FLOAT,
    total_charges FLOAT,
    churned INT  -- 0 or 1
);

-- Scoring table (new customers without labels)
CREATE TABLE customers_to_score (
    customer_id VARCHAR(20) PRIMARY KEY,
    tenure INT,
    monthly_charges FLOAT,
    total_charges FLOAT
);

-- Prediction output table (written by serving pipeline)
CREATE TABLE churn_predictions (
    customer_id VARCHAR(20),
    prediction INT,
    prediction_probability FLOAT,
    execution_date TIMESTAMP,
    model_run_id VARCHAR(100),
    serving_run_id VARCHAR(100)
);
```

### 3.2 SeaweedFS

```yaml
seaweedfs-master:
  image: chrislusf/seaweedfs
  command: "master -ip=seaweedfs-master -ip.bind=0.0.0.0"
  ports:
    - "9333:9333"

seaweedfs-volume:
  image: chrislusf/seaweedfs
  command: "volume -mserver=seaweedfs-master:9333 -ip.bind=0.0.0.0 -port=8080"
  depends_on:
    - seaweedfs-master

seaweedfs-filer:
  image: chrislusf/seaweedfs
  command: "filer -master=seaweedfs-master:9333 -ip.bind=0.0.0.0 -s3"
  ports:
    - "8333:8333"  # S3 API endpoint
    - "8888:8888"  # Filer HTTP
  depends_on:
    - seaweedfs-master
    - seaweedfs-volume
```

**Two buckets (created on startup via init script or `aws s3 mb`):**
- `mbt-mlflow-artifacts` -- MLflow artifact store (models, scalers, encoders)
- `mbt-pipeline-artifacts` -- Inter-step artifact storage for pod-per-step execution

```bash
# Create buckets after SeaweedFS is up
aws --endpoint-url http://localhost:8333 s3 mb s3://mbt-mlflow-artifacts
aws --endpoint-url http://localhost:8333 s3 mb s3://mbt-pipeline-artifacts
```

### 3.3 MLflow

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.12.1
  command: >
    mlflow server
    --backend-store-uri postgresql://mlflow_user:mlflow_password@postgres:5432/mlflow_db
    --default-artifact-root s3://mbt-mlflow-artifacts/
    --host 0.0.0.0
    --port 5000
  ports:
    - "5000:5000"
  environment:
    AWS_ACCESS_KEY_ID: seaweedfs_access_key
    AWS_SECRET_ACCESS_KEY: seaweedfs_secret_key
    MLFLOW_S3_ENDPOINT_URL: http://seaweedfs-filer:8333
  depends_on:
    - postgres
    - seaweedfs-filer
```

### 3.4 H2O Server

```yaml
h2o:
  image: h2oai/h2o-open-source-k8s:latest
  ports:
    - "54321:54321"
  environment:
    JAVA_OPTS: "-Xmx4g"
```

### 3.5 Gitea

```yaml
gitea:
  image: gitea/gitea:latest
  ports:
    - "3000:3000"
    - "2222:22"
  environment:
    GITEA__database__DB_TYPE: sqlite3
  volumes:
    - gitea_data:/data
```

**Setup after startup:**
- Create organization or user `de-team` (Data Engineer)
- Create user `ds-team` (Data Scientist)
- Create repository `de-team/mbt-demo-project`
- Add `ds-team` as collaborator with write access

### 3.6 Woodpecker CI

```yaml
woodpecker-server:
  image: woodpeckerci/woodpecker-server:latest
  ports:
    - "8000:8000"
  environment:
    WOODPECKER_GITEA: "true"
    WOODPECKER_GITEA_URL: http://gitea:3000
    WOODPECKER_GITEA_CLIENT: "<oauth2_client_id>"
    WOODPECKER_GITEA_SECRET: "<oauth2_secret>"
    WOODPECKER_SERVER_HOST: woodpecker-server:8000
    WOODPECKER_SECRET: "<woodpecker_secret>"
  depends_on:
    - gitea

woodpecker-agent:
  image: woodpeckerci/woodpecker-agent:latest
  environment:
    WOODPECKER_SERVER: woodpecker-server:9000
    WOODPECKER_SECRET: "<woodpecker_secret>"
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
  depends_on:
    - woodpecker-server
```

### 3.7 Airflow

```yaml
airflow-webserver:
  image: apache/airflow:2.8-python3.10
  command: webserver
  ports:
    - "8080:8080"
  environment:
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://airflow_user:airflow_password@postgres:5432/airflow_db
    AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
  volumes:
    - ./generated_dags:/opt/airflow/dags
  depends_on:
    - postgres

airflow-scheduler:
  image: apache/airflow:2.8-python3.10
  command: scheduler
  environment:
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://airflow_user:airflow_password@postgres:5432/airflow_db
    AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
  volumes:
    - ./generated_dags:/opt/airflow/dags
  depends_on:
    - postgres
```

Note: In production, Airflow would use `KubernetesExecutor` and `KubernetesPodOperator`. For local docker-compose testing, `LocalExecutor` with `BashOperator` calling `mbt step execute` can serve as a stand-in.

### 3.8 Zot Registry

```yaml
zot:
  image: ghcr.io/project-zot/zot-linux-amd64:latest
  ports:
    - "5050:5000"
  volumes:
    - zot_data:/var/lib/registry
```

### 3.9 Kubernetes (Local)

For local testing, use K3s or Kind:

```bash
# K3s (single-node)
curl -sfL https://get.k3s.io | sh -

# Or Kind
kind create cluster --name mbt-test
```

**Required K8s resources:**
- Namespace: `mbt-pipelines`
- ServiceAccount: `mbt-runner` (with image pull secret for Zot)
- ConfigMap: `mbt-config` (SeaweedFS credentials, PostgreSQL credentials, H2O URL, MLflow URI)
- Secret: `mbt-secrets` (passwords, S3 keys)

---

## 4. Profiles Configuration

The DE creates this `profiles.yaml` in the project root:

```yaml
demo-project:
  target: dev

  mlflow:
    experiment_name: churn_prediction

  orchestrator:
    type: airflow
    config:
      dag_directory: /opt/airflow/dags
      default_args:
        owner: de-team
        retries: 2
        retry_delay_minutes: 5

  outputs:
    dev:
      executor:
        type: local
      storage:
        type: local
        config:
          base_path: ./local_artifacts
      data_connector:
        type: local_file
        config:
          data_path: ./sample_data
      mlflow:
        tracking_uri: "file://./mlruns"
      secrets:
        provider: env

    prod:
      executor:
        type: kubernetes
        config:
          namespace: mbt-pipelines
          service_account: mbt-runner
          image: "zot-registry:5050/mbt-runner:latest"
      storage:
        type: s3
        config:
          bucket: mbt-pipeline-artifacts
          endpoint_url: "{{ env_var('SEAWEEDFS_ENDPOINT', 'http://seaweedfs-filer:8333') }}"
          access_key: "{{ secret('S3_ACCESS_KEY') }}"
          secret_key: "{{ secret('S3_SECRET_KEY') }}"
      data_connector:
        type: postgres
        config:
          host: "{{ env_var('PG_HOST', 'postgres') }}"
          port: 5432
          database: warehouse
          schema: public
          user: "{{ env_var('PG_USER', 'mbt_user') }}"
          password: "{{ secret('PG_PASSWORD') }}"
      mlflow:
        tracking_uri: "{{ env_var('MLFLOW_TRACKING_URI', 'http://mlflow:5000') }}"
        artifact_location: "s3://mbt-mlflow-artifacts/"
      h2o:
        url: "{{ env_var('H2O_URL', 'http://h2o:54321') }}"
      secrets:
        provider: env
```

- `dev` uses local file connector and local storage (works today with existing MBT)
- `prod` uses PostgreSQL, SeaweedFS S3, Kubernetes, and remote H2O

---

## 5. Pipeline YAML Examples

### 5.1 training_churn_model_v1.yaml

```yaml
schema_version: 1

project:
  name: training_churn_model_v1
  experiment_name: churn_prediction
  problem_type: binary_classification
  owner: ds-team
  tags: [churn, telecom, h2o, training]

deployment:
  mode: batch
  cadence: monthly

training:
  data_source:
    label_table: customers

  schema:
    target:
      label_column: churned
      classes: [0, 1]
      positive_class: 1
    identifiers:
      primary_key: customer_id

  model_training:
    framework: h2o_automl
    config:
      max_runtime_secs: 300
      max_models: 10
      sort_metric: AUC
      seed: 42
      nfolds: 5

  evaluation:
    primary_metric: roc_auc
    additional_metrics: [accuracy, f1, precision, recall]
    generate_plots: true
```

### 5.2 serving_churn_model_v1.yaml

```yaml
schema_version: 1

project:
  name: serving_churn_model_v1
  experiment_name: churn_prediction
  problem_type: binary_classification
  owner: ds-team
  tags: [churn, telecom, serving]

# Required for framework info and schema validation in apply_transforms
training:
  data_source:
    label_table: customers
  schema:
    target:
      label_column: churned
      classes: [0, 1]
      positive_class: 1
    identifiers:
      primary_key: customer_id
  model_training:
    framework: h2o_automl
    config: {}
  evaluation:
    primary_metric: roc_auc
    additional_metrics: []

deployment:
  mode: batch
  cadence: daily

serving:
  model_source:
    registry: mlflow
    run_id: "<run_id_from_mlflow>"  # DS hardcodes after reviewing MLflow UI
    artifact_snapshot: false

  data_source:
    scoring_table: customers_to_score

  output:
    destination: database        # NEW: requires publish.py to support write_table()
    table: churn_predictions
    include_probabilities: true
```

Note: `output.destination: database` is a new destination type. Currently `publish.py` only supports `local_file`. It must be extended to call `DataConnectorPlugin.write_table()` when destination is `database`.

---

## 6. Woodpecker CI Pipeline

`.woodpecker.yml` in the project root:

```yaml
steps:
  validate:
    image: zot-registry:5050/mbt-runner:latest
    commands:
      - mbt validate
    when:
      event: [push, pull_request]

  build-image:
    image: plugins/docker
    settings:
      repo: zot-registry:5050/mbt-runner
      registry: zot-registry:5050
      tags: latest
      dockerfile: Dockerfile
    when:
      event: push
      branch: master

  compile:
    image: zot-registry:5050/mbt-runner:latest
    commands:
      - |
        for pipeline in pipelines/*.yaml; do
          name=$(basename "$pipeline" .yaml)
          if [[ "$name" != _* ]]; then
            mbt compile "$name" --target prod
          fi
        done
    when:
      event: push
      branch: master

  generate-dags:
    image: zot-registry:5050/mbt-runner:latest
    commands:
      - mbt generate-dags --target prod --output ./generated_dags/
    when:
      event: push
      branch: master

  deploy-dags:
    image: zot-registry:5050/mbt-runner:latest
    commands:
      - aws --endpoint-url $SEAWEEDFS_ENDPOINT s3 sync ./generated_dags/ s3://mbt-pipeline-artifacts/dags/
      # Airflow DAG folder syncs from this S3 path, or use direct copy if volume-mounted
    when:
      event: push
      branch: master
    secrets: [S3_ACCESS_KEY, S3_SECRET_KEY, SEAWEEDFS_ENDPOINT]
```

**Pipeline stages:**
1. `validate` -- runs on every push and pull request, catches YAML errors early
2. `build-image` -- rebuilds the MBT Docker image and pushes to Zot (only on merge to master)
3. `compile` -- compiles all non-base pipelines with `--target prod`, generating manifests
4. `generate-dags` -- reads manifests, generates Airflow DAG Python files
5. `deploy-dags` -- deploys generated DAGs to where Airflow can pick them up

---

## 7. MBT Runner Dockerfile

```dockerfile
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc default-jre-headless curl \
    && rm -rf /var/lib/apt/lists/*

# Install MBT packages
COPY mbt-core/ /build/mbt-core/
COPY mbt-h2o/ /build/mbt-h2o/
COPY mbt-mlflow/ /build/mbt-mlflow/
COPY mbt-postgres/ /build/mbt-postgres/
COPY mbt-s3/ /build/mbt-s3/
COPY mbt-airflow/ /build/mbt-airflow/

RUN pip install --no-cache-dir \
    /build/mbt-core/ \
    /build/mbt-h2o/ \
    /build/mbt-mlflow/ \
    /build/mbt-postgres/ \
    /build/mbt-s3/ \
    /build/mbt-airflow/

# Copy project files (pipelines, profiles, compiled manifests)
COPY . /app
WORKDIR /app

ENTRYPOINT ["mbt"]
```

- `default-jre-headless` is required for H2O client JAR (even when connecting to remote server)
- `libpq-dev` + `gcc` needed for `psycopg2` compilation
- Image is pushed to Zot as `zot-registry:5050/mbt-runner:latest`

---

## 8. DE Workflow

The DE sets up infrastructure and the project before the DS begins.

### Step 1: Provision Infrastructure

```bash
docker-compose up -d
```

Verify all services are healthy:
```bash
docker-compose ps              # All services "Up"
psql -h localhost -U admin -c "SELECT 1"  # PostgreSQL
curl http://localhost:5000/health          # MLflow
curl http://localhost:54321/3/Cloud        # H2O
curl http://localhost:3000                 # Gitea
curl http://localhost:8080/health          # Airflow
```

### Step 2: Seed PostgreSQL

```bash
psql -h localhost -U admin -d warehouse -f infra/postgres/seed_data.sql
```

Verify: `SELECT COUNT(*) FROM customers;` returns rows.

### Step 3: Create SeaweedFS Buckets

```bash
aws --endpoint-url http://localhost:8333 s3 mb s3://mbt-mlflow-artifacts
aws --endpoint-url http://localhost:8333 s3 mb s3://mbt-pipeline-artifacts
```

### Step 4: Set Up Gitea Repository

1. Create `de-team` and `ds-team` accounts in Gitea UI
2. Create repository `de-team/mbt-demo-project`
3. Add `ds-team` as collaborator

### Step 5: Initialize MBT Project

```bash
mbt init demo-project
cd demo-project
```

### Step 6: Add DE-Owned Files

- `profiles.yaml` (Section 4)
- `.woodpecker.yml` (Section 6)
- `Dockerfile` (Section 7)
- `sample_data/customers.csv` and `sample_data/customers_to_score.csv` (for local dev)
- `.gitignore`:
  ```
  target/
  generated_dags/
  local_artifacts/
  mlruns/
  *.pyc
  __pycache__/
  ```

### Step 7: Build and Push Initial MBT Runner Image

```bash
docker build -t zot-registry:5050/mbt-runner:latest .
docker push zot-registry:5050/mbt-runner:latest
```

### Step 8: Configure Woodpecker CI

1. Enable the `mbt-demo-project` repo in Woodpecker UI
2. Add secrets: `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `SEAWEEDFS_ENDPOINT`, `PG_PASSWORD`

### Step 9: Push Initial Commit

```bash
git init
git remote add origin http://gitea:3000/de-team/mbt-demo-project.git
git add .
git commit -m "chore: Initial project setup with profiles and CI"
git push -u origin master
```

Verify: Woodpecker triggers. The `validate` step passes (no pipelines to validate yet, which is fine).

---

## 9. DS Workflow

### Phase A: Local Development

**Step 1: Clone the project**

```bash
git clone http://gitea:3000/de-team/mbt-demo-project.git
cd mbt-demo-project
```

Verify: `ls` shows `profiles.yaml`, `pipelines/`, `sample_data/`, `Dockerfile`, `.woodpecker.yml`

**Step 2: Create training pipeline YAML**

DS creates `pipelines/training_churn_model_v1.yaml` (see Section 5.1).

**Step 3: Validate the pipeline**

```bash
mbt validate training_churn_model_v1
```

Expected output: `Pipeline 'training_churn_model_v1' is valid.`

**Step 4: Compile locally**

```bash
mbt compile training_churn_model_v1 --target dev
```

Expected output: Manifest saved to `target/training_churn_model_v1/manifest.json`

Verify: `cat target/training_churn_model_v1/manifest.json` shows steps: `load_data`, `split_data`, `train_model`, `evaluate`, `log_run`

**Step 5: Run locally (generates run_id_1)**

```bash
mbt run --select training_churn_model_v1 --target dev
```

Expected output: All 5 steps pass. `run_id_1` (`run_YYYYMMDD_HHMMSS`) printed to console.

Verify:
- `local_artifacts/run_*/` directory exists with artifacts
- `target/training_churn_model_v1/run_results.json` shows `status: success`

Note: Local run uses `dev` target (local file connector, local storage, local H2O cluster). This tests the pipeline logic without infrastructure dependencies.

### Phase B: CI/CD and Production Training

**Step 6: Push branch and create merge request**

```bash
git checkout -b feature/add-training-pipeline
git add pipelines/training_churn_model_v1.yaml
git commit -m "feat: Add churn training pipeline v1"
git push origin feature/add-training-pipeline
```

DS creates a merge request in Gitea UI.

Verify: Woodpecker CI triggers. The `validate` step runs and passes.

**Step 7: Merge to master**

DE or DS merges the MR in Gitea. Woodpecker CI triggers the full pipeline:

1. `validate` -- passes
2. `build-image` -- rebuilds Docker image, pushes to Zot
3. `compile` -- generates `target/training_churn_model_v1/manifest.json` with `--target prod`
4. `generate-dags` -- generates `generated_dags/training_churn_model_v1_dag.py`
5. `deploy-dags` -- uploads DAG to Airflow's DAG folder

Verify:
- Woodpecker shows green for all 5 stages
- Zot registry has updated `mbt-runner:latest`
- Airflow UI shows DAG `training_churn_model_v1` (scheduled: monthly)

**Step 8: Trigger training pipeline first time (generates run_id_2)**

Manually trigger `training_churn_model_v1` DAG from Airflow UI (or wait for schedule).

Airflow launches K8s pods in sequence, one per step:

| Airflow Task | K8s Pod Command | Reads From | Writes To |
|---|---|---|---|
| `load_data` | `mbt step execute --pipeline training_churn_model_v1 --step load_data --target prod --run-id <id>` | PostgreSQL `customers` | SeaweedFS Bucket 2 |
| `split_data` | `mbt step execute ... --step split_data ...` | SeaweedFS (raw data) | SeaweedFS (train/test sets) |
| `train_model` | `mbt step execute ... --step train_model ...` | SeaweedFS (train set) + H2O Server | SeaweedFS (model) |
| `evaluate` | `mbt step execute ... --step evaluate ...` | SeaweedFS (model + test set) | SeaweedFS (metrics) |
| `log_run` | `mbt step execute ... --step log_run ...` | SeaweedFS (model + metrics) | MLflow -> SeaweedFS Bucket 1 + PostgreSQL `mlflow_db` |

Verify:
- Airflow DAG shows all tasks green
- MLflow UI shows a new run with metrics: `roc_auc`, `accuracy`, `f1`, `precision`, `recall`
- MLflow run has model artifact in SeaweedFS Bucket 1
- SeaweedFS Bucket 2 has artifacts under `run_<timestamp>/`

### Phase C: Model Selection and Serving Pipeline

**Step 9: DS reviews MLflow UI**

DS navigates to MLflow at `http://mlflow:5000`:
- Experiment: `churn_prediction`
- Compares `run_id_1` (local, dev) and `run_id_2` (Airflow, prod H2O AutoML)
- Picks the best `run_id` based on `roc_auc` metric
- Copies the chosen `run_id`

**Step 10: DS creates serving pipeline**

DS creates `pipelines/serving_churn_model_v1.yaml` (see Section 5.2), replacing `<run_id_from_mlflow>` with the actual chosen `run_id`.

**Step 11: DS tests serving locally**

```bash
mbt compile serving_churn_model_v1 --target dev
mbt run --select serving_churn_model_v1 --target dev
```

Verify: `predictions.csv` generated in project root with columns: `customer_id`, `prediction`, `prediction_probability`

### Phase D: Production Serving

**Step 12: DS pushes serving pipeline**

```bash
git checkout -b feature/add-serving-pipeline
git add pipelines/serving_churn_model_v1.yaml
git commit -m "feat: Add churn serving pipeline v1"
git push origin feature/add-serving-pipeline
```

Creates MR in Gitea. Woodpecker validates. MR merged to master.

Woodpecker full pipeline: validate -> build-image -> compile -> generate-dags -> deploy-dags

Verify: Airflow UI shows DAG `serving_churn_model_v1` (scheduled: daily)

**Step 13: Trigger serving pipeline (scores to PostgreSQL)**

Manually trigger `serving_churn_model_v1` DAG from Airflow UI (or wait for schedule).

Airflow launches K8s pods:

| Airflow Task | Reads From | Writes To |
|---|---|---|
| `load_scoring_data` | PostgreSQL `customers_to_score` | SeaweedFS Bucket 2 |
| `load_model` | MLflow -> SeaweedFS Bucket 1 | SeaweedFS Bucket 2 |
| `apply_transforms` | SeaweedFS (scoring data + model artifacts) | SeaweedFS (transformed data) |
| `predict` | SeaweedFS (model + transformed data) | SeaweedFS (predictions) |
| `publish` | SeaweedFS (predictions) | PostgreSQL `churn_predictions` |

Verify:
- All Airflow tasks green
- PostgreSQL: `SELECT COUNT(*) FROM churn_predictions;` returns rows
- PostgreSQL: `SELECT * FROM churn_predictions LIMIT 5;` shows valid predictions (0/1) and probabilities (0.0-1.0)

---

## 10. Pod-Per-Step Execution Architecture

### 10.1 The `mbt step execute` Command

Called by `KubernetesPodOperator` inside each pod:

```bash
mbt step execute \
    --pipeline training_churn_model_v1 \
    --step load_data \
    --target prod \
    --run-id run_20260301_120000
```

The `--run-id` is generated by the first step (or Airflow) and passed to all subsequent tasks via Airflow XCom or environment variable.

### 10.2 Execution Flow Inside a Pod

```
1. Load manifest from target/<pipeline>/manifest.json (baked into Docker image or fetched from S3)
2. Load profile config for the target
3. Initialize StoragePlugin (S3) from profile config
4. Load artifact registry from S3: s3://bucket/run_id/.artifact_registry.json
5. For each input declared in manifest.steps[step_name].inputs:
   a. Look up URI from artifact registry
   b. storage.get(uri) -> bytes
   c. Deserialize (pickle.loads)
6. Create RunContext with config, run_id, profile_config
7. Execute step.run(inputs, context) -> outputs
8. For each output:
   a. Serialize (pickle.dumps)
   b. storage.put(name, data, run_id, step_name) -> uri
   c. Update artifact registry with new URI
9. Save updated artifact registry to S3
```

### 10.3 Artifact Registry Persistence

The in-memory `artifact_registry` dict in `runner.py` must be persisted to S3 between pods:

- Location: `s3://mbt-pipeline-artifacts/<run_id>/.artifact_registry.json`
- Format: `{ "artifact_name": "s3://bucket/run_id/step/artifact" }`
- Each pod reads at start, updates after execution, writes back
- No concurrency issue: Airflow enforces DAG ordering, so steps never run in parallel unless they have no dependency

### 10.4 Manifest Distribution

Manifests are baked into the Docker image during the `build-image` CI step. Each image rebuild includes the latest compiled manifests in `target/`. Alternative: store manifests in SeaweedFS and fetch at pod startup for faster iteration without image rebuilds.

---

## 11. Verification Checklist

### End Goal 1: Model Viewable in MLflow

- [ ] Navigate to MLflow UI at `http://mlflow:5000`
- [ ] Experiment `churn_prediction` exists
- [ ] At least 2 runs visible:
  - `run_id_1` from local `mbt run` (dev target)
  - `run_id_2` from Airflow-triggered K8s execution (prod target)
- [ ] Production run (`run_id_2`) has:
  - Metrics: `roc_auc`, `accuracy`, `f1`, `precision`, `recall`
  - Parameters: `framework=h2o_automl`, `problem_type=binary_classification`
  - Artifacts: model directory in SeaweedFS Bucket 1
  - Tags: `pipeline_name=training_churn_model_v1`
- [ ] Model artifact downloadable from MLflow UI

### End Goal 2: Scores in PostgreSQL

- [ ] Connect: `psql -h postgres -U mbt_user -d warehouse`
- [ ] `SELECT COUNT(*) FROM churn_predictions;` returns rows matching `customers_to_score` count
- [ ] `SELECT * FROM churn_predictions LIMIT 5;` shows:
  - `customer_id` -- matches IDs from `customers_to_score`
  - `prediction` -- 0 or 1
  - `prediction_probability` -- between 0.0 and 1.0
  - `execution_date` -- timestamp of the serving run
  - `model_run_id` -- matches the `run_id` hardcoded in serving YAML
  - `serving_run_id` -- unique ID for this serving execution

### End Goal 3: Airflow DAGs Visible

- [ ] Airflow UI at `http://airflow:8080`
- [ ] DAG `training_churn_model_v1` exists with monthly schedule
- [ ] DAG `serving_churn_model_v1` exists with daily schedule
- [ ] Both DAGs show successful runs in their history

### Infrastructure Health

- [ ] K8s pods completed and cleaned up (no lingering pods in `mbt-pipelines` namespace)
- [ ] SeaweedFS Bucket 1 has model artifacts under `mbt-mlflow-artifacts/`
- [ ] SeaweedFS Bucket 2 has run artifacts under `mbt-pipeline-artifacts/<run_id>/`
- [ ] Woodpecker CI pipeline shows green for all stages
- [ ] Zot registry has the `mbt-runner:latest` image

---

## 12. Troubleshooting

| Symptom | Likely Cause | Debug Steps |
|---------|-------------|-------------|
| Pod fails with "artifact not found" | Artifact registry not persisted to S3 between pods | Check `.artifact_registry.json` in SeaweedFS Bucket 2. Verify previous step completed. |
| H2O connection timeout | H2O server unreachable from K8s pods | `kubectl exec -it <pod> -- curl http://h2o:54321/3/Cloud`. Check network policies. |
| MLflow logging fails | Wrong tracking URI or S3 credentials | Verify `MLFLOW_TRACKING_URI` and `AWS_*` env vars in pod. Check MLflow server logs. |
| PostgreSQL connection refused | Wrong credentials or network | `kubectl exec -it <pod> -- psql -h postgres -U mbt_user -d warehouse`. Check `pg_hba.conf`. |
| Woodpecker build fails | Docker socket or Zot access | Check agent logs. Verify Docker socket mount. Test `docker push` to Zot manually. |
| Airflow DAG not appearing | DAG file not deployed or has syntax error | Check `generated_dags/` contents. Run `python generated_dags/<file>.py` to check for import errors. |
| SeaweedFS S3 API errors | Wrong endpoint URL or missing buckets | `aws --endpoint-url <url> s3 ls`. Verify buckets exist. Check access keys. |
| Predictions not in PostgreSQL | `publish.py` doesn't support `destination: database` yet | Check if the `publish` step implementation handles `database` destination type. |

---

## 13. Appendix: SQL Seed Scripts

### init.sql (run by PostgreSQL on startup)

```sql
-- MLflow backend database
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

-- Airflow backend database
CREATE DATABASE airflow_db;
CREATE USER airflow_user WITH PASSWORD 'airflow_password';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

-- Warehouse database
CREATE DATABASE warehouse;
CREATE USER mbt_user WITH PASSWORD 'mbt_password';
GRANT ALL PRIVILEGES ON DATABASE warehouse TO mbt_user;

\connect warehouse;

-- Feature + label table
CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    tenure INT NOT NULL,
    monthly_charges FLOAT NOT NULL,
    total_charges FLOAT NOT NULL,
    churned INT NOT NULL CHECK (churned IN (0, 1))
);

-- Scoring table (no label column)
CREATE TABLE customers_to_score (
    customer_id VARCHAR(20) PRIMARY KEY,
    tenure INT NOT NULL,
    monthly_charges FLOAT NOT NULL,
    total_charges FLOAT NOT NULL
);

-- Prediction output table
CREATE TABLE churn_predictions (
    customer_id VARCHAR(20),
    prediction INT,
    prediction_probability FLOAT,
    execution_date TIMESTAMP DEFAULT NOW(),
    model_run_id VARCHAR(100),
    serving_run_id VARCHAR(100)
);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mbt_user;
```

### seed_data.sql (run by DE after startup)

```sql
\connect warehouse;

-- Seed historical customer data (matching the schema from examples/telecom-churn/sample_data/customers.csv)
INSERT INTO customers (customer_id, tenure, monthly_charges, total_charges, churned) VALUES
('CUST_00001', 12, 45.50, 546.00, 0),
('CUST_00002', 36, 89.20, 3211.20, 1),
('CUST_00003', 8, 35.75, 286.00, 0),
('CUST_00004', 48, 102.40, 4915.20, 0);
-- ... (continue with full dataset or use COPY FROM CSV)

-- Seed scoring data
INSERT INTO customers_to_score (customer_id, tenure, monthly_charges, total_charges) VALUES
('CUST_NEW_001', 12, 65.50, 786.00),
('CUST_NEW_002', 3, 45.20, 135.60),
('CUST_NEW_003', 24, 89.30, 2143.20),
('CUST_NEW_004', 6, 52.70, 316.20);
-- ... (continue with full dataset or use COPY FROM CSV)
```

Alternatively, load directly from CSVs:

```bash
psql -h localhost -U mbt_user -d warehouse \
    -c "\COPY customers FROM 'sample_data/customers.csv' WITH CSV HEADER"
psql -h localhost -U mbt_user -d warehouse \
    -c "\COPY customers_to_score FROM 'sample_data/customers_to_score.csv' WITH CSV HEADER"
```
