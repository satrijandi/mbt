Here's the step-by-step to run the E2E test manually:

Prerequisites
Docker and Docker Compose installed
uv sync already run from the repo root
Steps
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
Or run the automated script

cd ~/code/mbt/integration-test
bash run-test.sh
This does all of the above automatically (32 checks).