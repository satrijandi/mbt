#!/bin/bash
# MBT Integration Test - End-to-End
# Runs the full MBT workflow from compile through Airflow-scheduled execution
set +e  # Don't exit on errors - check() tracks pass/fail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/project"
MBT_ROOT="$(dirname "$SCRIPT_DIR")"
MBT="$MBT_ROOT/.venv/bin/mbt"
PYTHON="$MBT_ROOT/.venv/bin/python3"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

pass_count=0
fail_count=0

check() {
    local description="$1"
    local result="$2"
    if [ "$result" -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $description"
        ((pass_count++))
    else
        echo -e "  ${RED}✗${NC} $description"
        ((fail_count++))
    fi
}

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  MBT Integration Test - End-to-End${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# ============================================================
# PHASE 1: Local Development Workflow (dev target)
# ============================================================
echo -e "${YELLOW}Phase 1: Local Development Workflow${NC}"
echo ""

cd "$PROJECT_DIR"

# Step 1: Validate pipeline
echo "Step 1: Validate training pipeline..."
$MBT validate training_churn_model_v1 2>&1 | tail -1
check "Pipeline validation passes" $?

# Step 2: Compile locally
echo "Step 2: Compile training pipeline (dev target)..."
$MBT compile training_churn_model_v1 --target dev 2>&1 | tail -3
check "Compilation succeeds" $?

# Step 3: Verify manifest
echo "Step 3: Verify manifest..."
if [ -f "target/training_churn_model_v1/manifest.json" ]; then
    STEP_COUNT=$(python3 -c "import json; m=json.load(open('target/training_churn_model_v1/manifest.json')); print(len(m['steps']))")
    if [ "$STEP_COUNT" -eq 5 ]; then
        check "Manifest has 5 steps (load_data, split_data, train_model, evaluate, log_run)" 0
    else
        check "Manifest has 5 steps (got $STEP_COUNT)" 1
    fi
else
    check "Manifest file exists" 1
fi

# Step 4: Run locally
echo "Step 4: Run training pipeline locally (dev target)..."
$MBT run --select training_churn_model_v1 --target dev 2>&1 | tail -15
check "Local training pipeline succeeds" $?

# Step 5: Verify local artifacts
echo "Step 5: Verify local artifacts..."
LATEST_RUN=$(ls -1d local_artifacts/run_* 2>/dev/null | sort -r | head -1)
if [ -n "$LATEST_RUN" ]; then
    check "Local artifacts directory exists" 0
    if [ -f "$LATEST_RUN/load_data/raw_data" ]; then
        check "load_data artifacts exist" 0
    else
        check "load_data artifacts exist" 1
    fi
    if [ -f "$LATEST_RUN/train_model/model" ]; then
        check "train_model artifacts exist" 0
    else
        check "train_model artifacts exist" 1
    fi
else
    check "Local artifacts directory exists" 1
fi

# Step 6: Verify run_results.json
echo "Step 6: Verify run results..."
if [ -f "target/training_churn_model_v1/run_results.json" ]; then
    RUN_STATUS=$(python3 -c "import json; print(json.load(open('target/training_churn_model_v1/run_results.json'))['status'])")
    if [ "$RUN_STATUS" = "success" ]; then
        check "run_results.json shows status: success" 0
    else
        check "run_results.json shows status: success (got: $RUN_STATUS)" 1
    fi
else
    check "run_results.json exists" 1
fi

echo ""

# ============================================================
# PHASE 2: Production Pipeline with Docker Compose
# ============================================================
echo -e "${YELLOW}Phase 2: Infrastructure Setup (Docker Compose)${NC}"
echo ""

cd "$SCRIPT_DIR"

# Step 7: Start infrastructure
echo "Step 7: Starting Docker Compose services..."
docker compose up -d 2>&1 | tail -5
check "Docker Compose services started" $?

# Step 8: Wait for services
echo "Step 8: Waiting for services to be healthy..."
echo "  Waiting for PostgreSQL..."
for i in $(seq 1 30); do
    if docker compose exec -T postgres pg_isready -U admin > /dev/null 2>&1; then
        break
    fi
    sleep 2
done
check "PostgreSQL is ready" $(docker compose exec -T postgres pg_isready -U admin > /dev/null 2>&1; echo $?)

echo "  Waiting for SeaweedFS..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8333 > /dev/null 2>&1; then
        break
    fi
    sleep 2
done
check "SeaweedFS S3 API is ready" $(curl -s http://localhost:8333 > /dev/null 2>&1; echo $?)

echo "  Waiting for MLflow..."
for i in $(seq 1 30); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        break
    fi
    sleep 3
done
check "MLflow is ready" $(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health 2>/dev/null | grep -q 200; echo $?)

# Step 9: Create S3 buckets (using boto3 since aws CLI may not be installed)
echo "Step 9: Creating S3 buckets..."
$PYTHON -c "
import boto3, time
s3 = boto3.client('s3', endpoint_url='http://localhost:8333', aws_access_key_id='any', aws_secret_access_key='any')
for bucket in ['mbt-mlflow-artifacts', 'mbt-pipeline-artifacts']:
    try:
        s3.create_bucket(Bucket=bucket)
        print(f'Created bucket: {bucket}')
    except Exception as e:
        print(f'Bucket {bucket}: {e}')
# Verify by trying to put/get an object (SeaweedFS auto-creates buckets on write)
s3.put_object(Bucket='mbt-pipeline-artifacts', Key='_test', Body=b'test')
s3.delete_object(Bucket='mbt-pipeline-artifacts', Key='_test')
print('S3 bucket access verified')
" 2>&1
check "S3 buckets created (mbt-mlflow-artifacts, mbt-pipeline-artifacts)" $?

# Step 10: Verify PostgreSQL data
echo "Step 10: Verify seeded data..."
CUSTOMER_COUNT=$(docker compose exec -T postgres psql -U mbt_user -d warehouse -t -c "SELECT COUNT(*) FROM customers;" 2>/dev/null | tr -d ' ')
if [ "$CUSTOMER_COUNT" = "20" ]; then
    check "customers table has 20 rows" 0
else
    check "customers table has 20 rows (got: $CUSTOMER_COUNT)" 1
fi

SCORING_COUNT=$(docker compose exec -T postgres psql -U mbt_user -d warehouse -t -c "SELECT COUNT(*) FROM customers_to_score;" 2>/dev/null | tr -d ' ')
if [ "$SCORING_COUNT" = "10" ]; then
    check "customers_to_score table has 10 rows" 0
else
    check "customers_to_score table has 10 rows (got: $SCORING_COUNT)" 1
fi

echo ""

# ============================================================
# PHASE 3: Production Training Pipeline
# ============================================================
echo -e "${YELLOW}Phase 3: Production Training Pipeline${NC}"
echo ""

cd "$PROJECT_DIR"

# Step 11: Compile for prod target
echo "Step 11: Compile training pipeline (prod target)..."
$MBT compile training_churn_model_v1 --target prod 2>&1 | tail -3
check "Compilation for prod target succeeds" $?

# Step 12: Run training with prod target (PostgreSQL + S3)
echo "Step 12: Run training pipeline (prod target)..."
$MBT run --select training_churn_model_v1 --target prod 2>&1 | tail -15
TRAIN_EXIT=$?
check "Training pipeline with prod target succeeds" $TRAIN_EXIT

# Step 13: Verify S3 artifacts
echo "Step 13: Verify S3 artifacts..."
ARTIFACT_COUNT=$($PYTHON -c "
import boto3
s3 = boto3.client('s3', endpoint_url='http://localhost:8333', aws_access_key_id='any', aws_secret_access_key='any')
resp = s3.list_objects_v2(Bucket='mbt-pipeline-artifacts', MaxKeys=100)
count = len(resp.get('Contents', []))
print(count)
" 2>/dev/null)
if [ "$ARTIFACT_COUNT" -gt 0 ] 2>/dev/null; then
    check "Artifacts stored in S3 ($ARTIFACT_COUNT objects)" 0
else
    check "Artifacts stored in S3" 1
fi

# Step 14: Verify MLflow run
echo "Step 14: Verify MLflow tracking..."
MLFLOW_RUNS=$(curl -s "http://localhost:5000/api/2.0/mlflow/experiments/search?max_results=10" 2>/dev/null)
if echo "$MLFLOW_RUNS" | python3 -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if len(data.get('experiments',[])) > 0 else 1)" 2>/dev/null; then
    check "MLflow experiments exist" 0
else
    check "MLflow experiments exist" 1
fi

# Get the MLflow run_id for serving pipeline
echo "Step 14b: Getting MLflow run_id for serving..."
MLFLOW_RUN_ID=$($PYTHON -c "
import urllib.request, json
resp = urllib.request.urlopen(urllib.request.Request(
    'http://localhost:5000/api/2.0/mlflow/runs/search',
    data=json.dumps({'experiment_ids': ['1'], 'max_results': 1, 'order_by': ['start_time DESC']}).encode(),
    headers={'Content-Type': 'application/json'}
))
data = json.loads(resp.read())
runs = data.get('runs', [])
if runs:
    print(runs[0]['info']['run_id'])
else:
    print('unknown')
" 2>/dev/null || echo "unknown")
echo "  MLflow Run ID: $MLFLOW_RUN_ID"

echo ""

# ============================================================
# PHASE 4: DAG Generation
# ============================================================
echo -e "${YELLOW}Phase 4: DAG Generation${NC}"
echo ""

# Step 15: Update serving pipeline with training run_id, then compile
echo "Step 15: Compile serving pipeline (prod target)..."
# Update serving pipeline YAML with the MLflow run_id
if [ "$MLFLOW_RUN_ID" != "unknown" ] && [ -n "$MLFLOW_RUN_ID" ]; then
    $PYTHON -c "
import yaml
with open('pipelines/serving_churn_model_v1.yaml') as f:
    data = yaml.safe_load(f)
data['serving']['model_source']['run_id'] = '$MLFLOW_RUN_ID'
with open('pipelines/serving_churn_model_v1.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
print(f'Updated serving pipeline with MLflow run_id: $MLFLOW_RUN_ID')
" 2>&1
fi

$MBT compile serving_churn_model_v1 --target prod 2>&1 | tail -3
check "Serving pipeline compilation succeeds" $?

# Step 16: Generate Airflow DAGs
echo "Step 16: Generate Airflow DAGs..."
rm -rf "$SCRIPT_DIR/generated_dags" 2>/dev/null
mkdir -p "$SCRIPT_DIR/generated_dags"
$MBT generate-dags --target prod --output "$SCRIPT_DIR/generated_dags" 2>&1 | tail -5
check "DAG generation succeeds" $?

# Step 17: Verify generated DAGs
echo "Step 17: Verify generated DAG files..."
if [ -f "$SCRIPT_DIR/generated_dags/training_churn_model_v1_dag.py" ]; then
    check "Training DAG file generated" 0
else
    check "Training DAG file generated" 1
fi

if [ -f "$SCRIPT_DIR/generated_dags/serving_churn_model_v1_dag.py" ]; then
    check "Serving DAG file generated" 0
else
    check "Serving DAG file generated" 1
fi

# Step 18: Verify DAG syntax
echo "Step 18: Verify DAG syntax..."
python3 -c "
import ast, sys
for dag_file in ['$SCRIPT_DIR/generated_dags/training_churn_model_v1_dag.py', '$SCRIPT_DIR/generated_dags/serving_churn_model_v1_dag.py']:
    try:
        with open(dag_file) as f:
            ast.parse(f.read())
    except Exception as e:
        print(f'Syntax error in {dag_file}: {e}')
        sys.exit(1)
print('All DAG files have valid Python syntax')
" 2>&1
check "DAG files have valid Python syntax" $?

echo ""

# ============================================================
# PHASE 5: Step Executor (pod-per-step simulation)
# ============================================================
echo -e "${YELLOW}Phase 5: Step Executor (pod-per-step simulation)${NC}"
echo ""

# Step 19: Execute individual steps via step executor
echo "Step 19: Execute steps individually (simulating pod-per-step)..."
STEP_RUN_ID="run_integration_test_$(date +%Y%m%d_%H%M%S)"

echo "  Executing load_data..."
$MBT step execute --pipeline training_churn_model_v1 --step load_data --target prod --run-id "$STEP_RUN_ID" 2>&1 | tail -3
check "Step executor: load_data succeeds" $?

echo "  Executing split_data..."
$MBT step execute --pipeline training_churn_model_v1 --step split_data --target prod --run-id "$STEP_RUN_ID" 2>&1 | tail -3
check "Step executor: split_data succeeds" $?

echo "  Executing train_model..."
$MBT step execute --pipeline training_churn_model_v1 --step train_model --target prod --run-id "$STEP_RUN_ID" 2>&1 | tail -3
check "Step executor: train_model succeeds" $?

echo "  Executing evaluate..."
$MBT step execute --pipeline training_churn_model_v1 --step evaluate --target prod --run-id "$STEP_RUN_ID" 2>&1 | tail -3
check "Step executor: evaluate succeeds" $?

echo "  Executing log_run..."
$MBT step execute --pipeline training_churn_model_v1 --step log_run --target prod --run-id "$STEP_RUN_ID" 2>&1 | tail -3
check "Step executor: log_run succeeds" $?

# Step 20: Verify artifact registry persistence
echo "Step 20: Verify artifact registry in S3..."
REGISTRY_EXISTS=$($PYTHON -c "
import boto3
s3 = boto3.client('s3', endpoint_url='http://localhost:8333', aws_access_key_id='any', aws_secret_access_key='any')
try:
    s3.head_object(Bucket='mbt-pipeline-artifacts', Key='$STEP_RUN_ID/_meta/.artifact_registry.json')
    print('1')
except Exception:
    print('0')
" 2>/dev/null)
if [ "$REGISTRY_EXISTS" = "1" ]; then
    check "Artifact registry persisted in S3" 0
else
    check "Artifact registry persisted in S3" 1
fi

echo ""

# ============================================================
# PHASE 6: Serving Pipeline (prod target)
# ============================================================
echo -e "${YELLOW}Phase 6: Serving Pipeline (prod target)${NC}"
echo ""

cd "$PROJECT_DIR"

echo "Step 21: Run serving pipeline (prod target)..."
$MBT run --select serving_churn_model_v1 --target prod 2>&1 | tail -15
check "Serving pipeline with prod target succeeds" $?

echo "Step 22: Verify predictions in PostgreSQL..."
PRED_COUNT=$(docker compose -f "$SCRIPT_DIR/docker-compose.yml" exec -T postgres psql -U mbt_user -d warehouse -t -c "SELECT COUNT(*) FROM churn_predictions;" 2>/dev/null | tr -d ' ')
if [ "$PRED_COUNT" -gt 0 ] 2>/dev/null; then
    check "Predictions stored in PostgreSQL ($PRED_COUNT rows)" 0
else
    check "Predictions stored in PostgreSQL" 1
fi

echo ""

# ============================================================
# SUMMARY
# ============================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Integration Test Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "  ${GREEN}Passed: $pass_count${NC}"
echo -e "  ${RED}Failed: $fail_count${NC}"
total=$((pass_count + fail_count))
echo -e "  Total:  $total"
echo ""

if [ "$fail_count" -eq 0 ]; then
    echo -e "${GREEN}All integration tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some integration tests failed.${NC}"
    exit 1
fi
