# MBT Integration Tests

End-to-end integration tests for MBT with PostgreSQL, S3 (SeaweedFS), and MLflow.

## Available Test Pipelines

The integration test project includes three training pipelines to test different feature levels:

### training_churn_model_v1.yaml (Basic)
- **Purpose:** Backward compatibility test
- **Features:** Single table, simple random split, sklearn RandomForest
- **Data:** Original `customers.csv` (20 rows)
- **Use for:** Quick smoke test, backward compatibility verification

### training_churn_model_v2.yaml (Typical DS Pattern)
- **Purpose:** Test multi-table joins and temporal windowing
- **Features:**
  - Multi-table joins (label_table + features_table_a + features_table_b)
  - Temporal windowing (9 months train, 1 month test)
  - Drift detection with PSI
- **Data:** Synthetic data (200 customers, 100 features, 365 days)
- **Use for:** Testing production-like workflows

### training_churn_model_v3.yaml (Full Advanced)
- **Purpose:** Test all advanced features including feature selection
- **Features:**
  - Multi-table joins with 100+ features
  - Temporal windowing
  - Feature selection (variance threshold + correlation + LGBM importance)
  - Drift detection with PSI
- **Data:** Same as v2 but with feature selection enabled
- **Use for:** Testing complete typical DS pipeline pattern
- **Note:** Requires `lightgbm` installed: `pip install lightgbm`

## Test Data

The integration test includes both simple and realistic synthetic data:

- **Basic data:** `customers.csv` (20 rows, 5 columns) - for v1
- **Advanced data:** Multi-table setup (generated):
  - `label_table.csv` - 54,600 rows with customer_id, snapshot_date, is_churn
  - `features_table_a.csv` - 73,000 rows with 50 features
  - `features_table_b.csv` - 73,000 rows with 50 features
  - `customers_to_score.csv` - 2,400 rows for serving

Data includes realistic characteristics:
- 15% missing values
- 5% constant features
- Highly correlated feature pairs
- Temporal patterns across 12 months (2025-01-01 to 2025-12-31)

## Running Tests

Here's the step-by-step to run the E2E test manually:

### Prerequisites
- Docker and Docker Compose installed
- `uv sync` already run from the repo root
- Optional: `pip install lightgbm` for v3 pipeline

### Steps
1. Start infrastructure


cd ~/code/mbt/integration-test
docker compose up -d --build
2. Wait for services (PostgreSQL, SeaweedFS, MLflow)


# PostgreSQL
docker compose exec -T postgres pg_isready -U admin

# SeaweedFS
curl -s http://localhost:8333

# MLflow (may take ~10s for DB migrations)
curl -s http://localhost:5000/health
3. Create S3 buckets


.venv/bin/python3 -c "
import boto3
s3 = boto3.client('s3', endpoint_url='http://localhost:8333', aws_access_key_id='any', aws_secret_access_key='any')
s3.create_bucket(Bucket='mbt-mlflow-artifacts')
s3.create_bucket(Bucket='mbt-pipeline-artifacts')
"
(run from repo root so .venv resolves)

4. Compile and run training pipeline (prod target)


cd ~/code/mbt/integration-test/project
../../.venv/bin/mbt compile training_churn_model_v1 --target prod
../../.venv/bin/mbt run --select training_churn_model_v1 --target prod

# OR test the advanced pipeline (v2 with multi-table joins)
../../.venv/bin/mbt compile training_churn_model_v2 --target prod
../../.venv/bin/mbt run --select training_churn_model_v2 --target prod

# OR test the full pipeline (v3 with feature selection)
# Requires: pip install lightgbm
../../.venv/bin/mbt compile training_churn_model_v3 --target prod
../../.venv/bin/mbt run --select training_churn_model_v3 --target prod
5. Get the MLflow run ID and update serving pipeline


# Get MLflow run ID from API
curl -s http://localhost:5000/api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids":["1"],"max_results":1}' | python3 -m json.tool
Copy the run_id value, then edit serving_churn_model_v1.yaml and replace the run_id field with it.

6. Compile and run serving pipeline


../../.venv/bin/mbt compile serving_churn_model_v1 --target prod
../../.venv/bin/mbt run --select serving_churn_model_v1 --target prod
7. Verify predictions in PostgreSQL


cd ~/code/mbt/integration-test
docker compose exec -T postgres psql -U mbt_user -d warehouse \
  -c "SELECT * FROM churn_predictions;"
8. Generate Airflow DAGs


cd ~/code/mbt/integration-test/project
mkdir -p ../generated_dags
../../.venv/bin/mbt generate-dags --target prod --output ../generated_dags
9. Test step executor (pod-per-step simulation)


RUN_ID="run_manual_test_$(date +%Y%m%d_%H%M%S)"
../../.venv/bin/mbt step execute --pipeline training_churn_model_v1 --step load_data --target prod --run-id "$RUN_ID"
../../.venv/bin/mbt step execute --pipeline training_churn_model_v1 --step split_data --target prod --run-id "$RUN_ID"
../../.venv/bin/mbt step execute --pipeline training_churn_model_v1 --step train_model --target prod --run-id "$RUN_ID"
../../.venv/bin/mbt step execute --pipeline training_churn_model_v1 --step evaluate --target prod --run-id "$RUN_ID"
../../.venv/bin/mbt step execute --pipeline training_churn_model_v1 --step log_run --target prod --run-id "$RUN_ID"
10. Clean up when done


cd ~/code/mbt/integration-test
docker compose down -v

## Verifying Advanced Pipeline Features

When testing v2 or v3 pipelines, verify these additional features:

### Multi-Table Joins
Check logs for join operations:
```bash
grep "Joining.*feature table" integration-test/project/local_artifacts/run_*/logs/load_data.log
```

Expected output should show:
- Loaded label table: X rows, 3 columns
- Joining 2 feature table(s)...
- features_table_a: Y rows, 52 columns
- features_table_b: Z rows, 52 columns
- Final joined data: X rows, 103 columns

### Temporal Windowing
Check split logs for temporal windows:
```bash
grep "Temporal windows" integration-test/project/local_artifacts/run_*/logs/split_data.log
```

Expected output:
- Train: 2025-01-01 to 2025-10-01 (9 months)
- Test: 2025-10-01 to 2025-11-01 (1 month)

### Drift Detection (v2 and v3)
Check if drift_info artifact was created:
```bash
ls -lh integration-test/project/local_artifacts/run_*/drift_info.csv
head integration-test/project/local_artifacts/run_*/drift_info.csv
```

Expected: CSV with columns `feature, period, drift_method, drift_value`

### Feature Selection (v3 only)
Check feature selection logs:
```bash
grep "LGBM" integration-test/project/local_artifacts/run_*/logs/feature_selection.log
```

Expected output showing:
- Variance threshold: removed X features
- Correlation filter: removed Y features
- LGBM importance: kept Z features

## Automated Test Script

Run all checks automatically:

```bash
cd ~/code/mbt/integration-test
bash run-test.sh
```

This does all of the above automatically (32 checks).

## Quick Test of New Features (Local)

Test the new features locally without Docker infrastructure:

```bash
cd integration-test/project

# Test v2 (multi-table + temporal windowing + drift)
../../.venv/bin/mbt compile training_churn_model_v2 --target dev
../../.venv/bin/mbt run --select training_churn_model_v2 --target dev

# Test v3 (with feature selection)
../../.venv/bin/mbt compile training_churn_model_v3 --target dev
../../.venv/bin/mbt run --select training_churn_model_v3 --target dev

# Verify outputs
ls -lh local_artifacts/run_*/
cat local_artifacts/run_*/metrics.json
head local_artifacts/run_*/drift_info.csv
```