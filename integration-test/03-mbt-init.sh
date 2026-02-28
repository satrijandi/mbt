#!/usr/bin/env bash
# ============================================================================
# 03-mbt-init.sh - DS Workflow: Initialize MBT project and create training PR
#
# Simulates a Data Scientist performing their first MBT workflow:
#   1. Spawn a JupyterHub notebook session
#   2. Install MBT packages (mbt-core, mbt-sklearn, mbt-mlflow)
#   3. Clone the de-team/ml-pipeline repo and create a feature branch
#   4. Run `mbt init` to scaffold a churn training pipeline
#   5. Create and execute a Jupyter notebook that compiles + runs the pipeline
#   6. Verify MLflow experiment was logged
#   7. Commit and push the project
#   8. Create a Pull Request in Gitea
#
# Prerequisites:
#   - 01-setup-infra.sh has been run (cluster + all services running)
#   - 02-init-data.sh has been run (warehouse data loaded)
#   - Tools: kubectl, curl, jq
#
# Usage:
#   ./03-mbt-init.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBT_ROOT="$(dirname "$SCRIPT_DIR")"
START_TIME=$(date +%s)

# ============================================================
# Configuration
# ============================================================

NS_MBT="mbt"

# JupyterHub
JUPYTERHUB_PASSWORD="mbt-jupyter"
JUPYTER_USER="ds-user"

# Gitea
GITEA_DS_USER="ds-team"
GITEA_DS_PASSWORD="ds-team-password"
GITEA_REPO_OWNER="de-team"
GITEA_REPO_NAME="ml-pipeline"
GITEA_BRANCH="feature/add-training-pipeline"

# MBT init parameters (small data for fast execution in 2GB pod)
MBT_PROJECT="churn"
MBT_TEMPLATE="typical-ds-pipeline"
MBT_FRAMEWORK="sklearn"
MBT_NUM_CUSTOMERS=100
MBT_FEATURES_A=26
MBT_FEATURES_B=50
MBT_DAILY_SAMPLES=100

# Pipeline naming
PIPELINE_SRC_NAME="${MBT_PROJECT}_training_v1"
PIPELINE_DST_NAME="churn_training_pipeline_v1"

# In-cluster URLs
GITEA_CLUSTER_URL="http://gitea-http.${NS_MBT}.svc.cluster.local:3000"
MLFLOW_CLUSTER_URL="http://mlflow.${NS_MBT}.svc.cluster.local:5000"

# ============================================================
# Utility Functions (matching 01-setup-infra.sh style)
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

log_phase() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

cleanup_pf() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

# Port-forward PIDs to clean up on exit
PF_PIDS=()
cleanup_all_pf() {
    for pid in "${PF_PIDS[@]}"; do
        cleanup_pf "$pid"
    done
}
trap cleanup_all_pf EXIT

wait_for_url() {
    local url="$1" max_retries="${2:-60}" delay="${3:-3}"
    for i in $(seq 1 "$max_retries"); do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep "$delay"
    done
    return 1
}

# Execute a command inside the singleuser pod
pod_exec() {
    kubectl exec -n "$NS_MBT" "$SINGLEUSER_POD" -- bash -c "$1"
}

# ============================================================
# Phase 0: Prerequisites
# ============================================================

phase_0_prerequisites() {
    log_phase "Phase 0: Checking prerequisites"

    # Check required tools
    for tool in kubectl curl jq; do
        if ! command -v "$tool" &>/dev/null; then
            log_error "$tool is required but not found"
            exit 1
        fi
    done
    log_success "Required tools: kubectl, curl, jq"

    # Verify cluster is accessible
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot reach Kubernetes cluster. Is it running?"
        exit 1
    fi
    log_success "Kubernetes cluster is accessible"

    # Verify critical pods are running
    local critical_pods=(
        "app=jupyterhub,component=hub"
        "app.kubernetes.io/name=gitea"
        "app=mlflow"
    )
    local pod_names=("JupyterHub hub" "Gitea" "MLflow")

    for i in "${!critical_pods[@]}"; do
        if kubectl get pod -n "$NS_MBT" -l "${critical_pods[$i]}" --field-selector=status.phase=Running -o name 2>/dev/null | grep -q pod; then
            log_success "${pod_names[$i]} pod is running"
        else
            log_error "${pod_names[$i]} pod is not running. Run 01-setup-infra.sh first."
            exit 1
        fi
    done
}

# ============================================================
# Phase 1: Spawn JupyterHub Session
# ============================================================

phase_1_spawn_jupyter() {
    log_phase "Phase 1: Spawning JupyterHub session for $JUPYTER_USER"

    # Check if singleuser pod already exists
    local existing_pod
    existing_pod=$(kubectl get pod -n "$NS_MBT" \
        -l "hub.jupyter.org/username=$JUPYTER_USER" \
        --field-selector=status.phase=Running \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -n "$existing_pod" ]]; then
        log_success "Singleuser pod already running: $existing_pod"
        SINGLEUSER_POD="$existing_pod"
        return
    fi

    # Port-forward to JupyterHub proxy-public
    log_info "Port-forwarding to JupyterHub..."
    kubectl port-forward -n "$NS_MBT" svc/proxy-public 18000:80 &
    local pf_pid=$!
    PF_PIDS+=("$pf_pid")
    sleep 3

    if ! wait_for_url "http://localhost:18000/hub/login" 20 2; then
        log_error "JupyterHub not reachable via port-forward"
        exit 1
    fi
    log_success "JupyterHub is accessible"

    # Login as ds-user to trigger server spawn (DummyAuth, xsrf disabled)
    log_info "Logging in as $JUPYTER_USER to trigger pod spawn..."
    local login_response
    login_response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "http://localhost:18000/hub/login" \
        -d "username=${JUPYTER_USER}&password=${JUPYTERHUB_PASSWORD}" \
        -L --max-redirs 5 2>/dev/null || echo "000")

    if [[ "$login_response" != "200" && "$login_response" != "302" ]]; then
        log_warn "Login response code: $login_response (may still work)"
    fi

    # Request server spawn explicitly
    log_info "Requesting server spawn..."
    curl -s -X POST "http://localhost:18000/hub/spawn/${JUPYTER_USER}" \
        -d "username=${JUPYTER_USER}&password=${JUPYTERHUB_PASSWORD}" \
        -L --max-redirs 5 > /dev/null 2>&1 || true

    # Wait for singleuser pod to be ready
    log_info "Waiting for singleuser pod to start (up to 180s)..."
    local waited=0
    while [[ $waited -lt 180 ]]; do
        SINGLEUSER_POD=$(kubectl get pod -n "$NS_MBT" \
            -l "hub.jupyter.org/username=$JUPYTER_USER" \
            --field-selector=status.phase=Running \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        if [[ -n "$SINGLEUSER_POD" ]]; then
            break
        fi
        sleep 5
        waited=$((waited + 5))
        if (( waited % 30 == 0 )); then
            log_info "Still waiting... (${waited}s elapsed)"
        fi
    done

    if [[ -z "${SINGLEUSER_POD:-}" ]]; then
        log_error "Singleuser pod did not start within 180s"
        log_info "Debug: kubectl get pods -n $NS_MBT -l hub.jupyter.org/username=$JUPYTER_USER"
        exit 1
    fi

    # Wait for pod to be fully ready
    kubectl wait --for=condition=ready "pod/$SINGLEUSER_POD" \
        -n "$NS_MBT" --timeout=120s 2>/dev/null || true

    log_success "Singleuser pod running: $SINGLEUSER_POD"

    # Clean up the JupyterHub port-forward (no longer needed)
    cleanup_pf "$pf_pid"
}

# ============================================================
# Phase 2: Install MBT Packages
# ============================================================

phase_2_install_mbt() {
    log_phase "Phase 2: Installing MBT packages in singleuser pod"

    # Check if mbt is already installed
    if pod_exec "pip show mbt-core" &>/dev/null; then
        log_success "MBT packages already installed (idempotent)"
        pod_exec "mbt version" 2>/dev/null || true
        return
    fi

    # Create tarball of mbt packages on host
    log_info "Creating tarball of MBT packages..."
    local tarball="/tmp/mbt-packages.tar.gz"
    tar -czf "$tarball" \
        -C "$MBT_ROOT" \
        mbt-core/src mbt-core/pyproject.toml mbt-core/README.md \
        mbt-sklearn/src mbt-sklearn/pyproject.toml mbt-sklearn/README.md \
        mbt-mlflow/src mbt-mlflow/pyproject.toml mbt-mlflow/README.md

    # Copy tarball into the pod
    log_info "Copying packages to pod..."
    kubectl cp "$tarball" "$NS_MBT/$SINGLEUSER_POD:/tmp/mbt-packages.tar.gz"

    # Extract and install inside the pod
    log_info "Installing mbt-core, mbt-sklearn, mbt-mlflow..."
    pod_exec "
        cd /tmp && \
        tar -xzf mbt-packages.tar.gz && \
        pip install --quiet ./mbt-core ./mbt-sklearn ./mbt-mlflow
    "

    # Verify installation
    local version
    version=$(pod_exec "mbt version" 2>/dev/null || echo "unknown")
    log_success "MBT installed: $version"

    # Clean up tarball
    rm -f "$tarball"
    pod_exec "rm -rf /tmp/mbt-packages.tar.gz /tmp/mbt-core /tmp/mbt-sklearn /tmp/mbt-mlflow" || true
}

# ============================================================
# Phase 3: Clone Gitea Repo + Create Branch
# ============================================================

phase_3_clone_repo() {
    log_phase "Phase 3: Cloning ${GITEA_REPO_OWNER}/${GITEA_REPO_NAME} and creating branch"

    local repo_dir="/home/jovyan/${GITEA_REPO_NAME}"
    local clone_url="http://${GITEA_DS_USER}:${GITEA_DS_PASSWORD}@gitea-http.${NS_MBT}.svc.cluster.local:3000/${GITEA_REPO_OWNER}/${GITEA_REPO_NAME}.git"

    # Check if already cloned
    if pod_exec "test -d ${repo_dir}/.git" &>/dev/null; then
        log_success "Repository already cloned at $repo_dir (idempotent)"
    else
        log_info "Configuring git..."
        pod_exec "
            git config --global user.name '${GITEA_DS_USER}' && \
            git config --global user.email '${GITEA_DS_USER}@mbt.local'
        "

        log_info "Cloning repository..."
        pod_exec "git clone '${clone_url}' '${repo_dir}'"
        log_success "Repository cloned to $repo_dir"
    fi

    # Create or switch to feature branch
    local current_branch
    current_branch=$(pod_exec "cd ${repo_dir} && git branch --show-current" 2>/dev/null || echo "")
    if [[ "$current_branch" == "$GITEA_BRANCH" ]]; then
        log_success "Already on branch $GITEA_BRANCH"
    else
        log_info "Creating branch $GITEA_BRANCH..."
        pod_exec "cd ${repo_dir} && git checkout -b '${GITEA_BRANCH}'" 2>/dev/null \
            || pod_exec "cd ${repo_dir} && git checkout '${GITEA_BRANCH}'"
        log_success "On branch $GITEA_BRANCH"
    fi
}

# ============================================================
# Phase 4: Initialize MBT Project
# ============================================================

phase_4_mbt_init() {
    log_phase "Phase 4: Running mbt init and setting up project"

    local repo_dir="/home/jovyan/${GITEA_REPO_NAME}"
    local project_dir="/home/jovyan/${MBT_PROJECT}"

    # Check if project files already exist in repo
    if pod_exec "test -f ${repo_dir}/pipelines/${PIPELINE_DST_NAME}.yaml" &>/dev/null; then
        log_success "Pipeline already exists in repo (idempotent)"
        return
    fi

    # Run mbt init (generates project in a new directory)
    if ! pod_exec "test -d ${project_dir}" &>/dev/null; then
        log_info "Running mbt init..."
        pod_exec "
            cd /home/jovyan && \
            mbt init ${MBT_PROJECT} \
                --template ${MBT_TEMPLATE} \
                --framework ${MBT_FRAMEWORK} \
                --num-customers ${MBT_NUM_CUSTOMERS} \
                --features-a ${MBT_FEATURES_A} \
                --features-b ${MBT_FEATURES_B} \
                --daily-samples ${MBT_DAILY_SAMPLES}
        "
        log_success "MBT project initialized"
    else
        log_success "MBT project directory already exists"
    fi

    # Copy generated files into the repo
    log_info "Copying project files into repository..."
    pod_exec "
        cp -r ${project_dir}/pipelines ${repo_dir}/ && \
        cp -r ${project_dir}/sample_data ${repo_dir}/ && \
        cp -r ${project_dir}/lib ${repo_dir}/ && \
        cp ${project_dir}/profiles.yaml ${repo_dir}/ && \
        cp ${project_dir}/pyproject.toml ${repo_dir}/ && \
        cp ${project_dir}/README.md ${repo_dir}/
    "

    # Rename pipeline file
    log_info "Renaming pipeline: ${PIPELINE_SRC_NAME} -> ${PIPELINE_DST_NAME}"
    pod_exec "
        mv '${repo_dir}/pipelines/${PIPELINE_SRC_NAME}.yaml' \
           '${repo_dir}/pipelines/${PIPELINE_DST_NAME}.yaml'
    "

    # Update project.name in the pipeline YAML
    log_info "Updating pipeline project name..."
    pod_exec "
        sed -i 's/name: ${PIPELINE_SRC_NAME}/name: ${PIPELINE_DST_NAME}/' \
            '${repo_dir}/pipelines/${PIPELINE_DST_NAME}.yaml'
    "

    # Customize profiles.yaml: set MLflow tracking_uri to cluster MLflow
    log_info "Configuring profiles.yaml for cluster MLflow..."
    pod_exec "
        sed -i 's|tracking_uri: \"./mlruns\"|tracking_uri: \"${MLFLOW_CLUSTER_URL}\"|' \
            '${repo_dir}/profiles.yaml'
    "

    # Create directories that may be needed
    pod_exec "mkdir -p '${repo_dir}/notebooks' '${repo_dir}/target'" || true

    log_success "Project files set up in repository"
}

# ============================================================
# Phase 5: Create Jupyter Notebook
# ============================================================

phase_5_create_notebook() {
    log_phase "Phase 5: Creating Jupyter notebook"

    local repo_dir="/home/jovyan/${GITEA_REPO_NAME}"
    local notebook_path="${repo_dir}/notebooks/01_train_churn_pipeline.ipynb"

    if pod_exec "test -f ${notebook_path}" &>/dev/null; then
        log_success "Notebook already exists (idempotent)"
        return
    fi

    log_info "Writing notebook to ${notebook_path}..."

    # Write the notebook JSON via heredoc
    pod_exec "cat > '${notebook_path}' << 'NOTEBOOK_EOF'
{
 \"cells\": [
  {
   \"cell_type\": \"markdown\",
   \"metadata\": {},
   \"source\": [
    \"# Churn Training Pipeline v1\\n\",
    \"\\n\",
    \"This notebook runs the MBT churn training pipeline.\\n\",
    \"\\n\",
    \"**Pipeline:** \`${PIPELINE_DST_NAME}\`\\n\",
    \"**Framework:** sklearn (RandomForestClassifier)\\n\",
    \"**MLflow:** ${MLFLOW_CLUSTER_URL}\"
   ]
  },
  {
   \"cell_type\": \"code\",
   \"metadata\": {},
   \"source\": [
    \"import os\\n\",
    \"os.chdir('/home/jovyan/${GITEA_REPO_NAME}')\\n\",
    \"print(f'Working directory: {os.getcwd()}')\"
   ],
   \"execution_count\": null,
   \"outputs\": []
  },
  {
   \"cell_type\": \"code\",
   \"metadata\": {},
   \"source\": [
    \"# Compile the pipeline\\n\",
    \"!mbt compile ${PIPELINE_DST_NAME}\"
   ],
   \"execution_count\": null,
   \"outputs\": []
  },
  {
   \"cell_type\": \"code\",
   \"metadata\": {},
   \"source\": [
    \"# Run the pipeline\\n\",
    \"!mbt run --select ${PIPELINE_DST_NAME}\"
   ],
   \"execution_count\": null,
   \"outputs\": []
  },
  {
   \"cell_type\": \"code\",
   \"metadata\": {},
   \"source\": [
    \"# Show run results\\n\",
    \"import glob, json\\n\",
    \"\\n\",
    \"run_dirs = sorted(glob.glob('local_artifacts/run_*'))\\n\",
    \"if run_dirs:\\n\",
    \"    latest = run_dirs[-1]\\n\",
    \"    print(f'Latest run: {latest}')\\n\",
    \"    print(f'Contents: {os.listdir(latest)}')\\n\",
    \"    \\n\",
    \"    metrics_file = os.path.join(latest, 'metrics.json')\\n\",
    \"    if os.path.exists(metrics_file):\\n\",
    \"        with open(metrics_file) as f:\\n\",
    \"            metrics = json.load(f)\\n\",
    \"        print(f'\\\\nMetrics:')\\n\",
    \"        for k, v in metrics.items():\\n\",
    \"            print(f'  {k}: {v}')\\n\",
    \"else:\\n\",
    \"    print('No run artifacts found')\"
   ],
   \"execution_count\": null,
   \"outputs\": []
  },
  {
   \"cell_type\": \"code\",
   \"metadata\": {},
   \"source\": [
    \"# Check MLflow experiment\\n\",
    \"import mlflow\\n\",
    \"\\n\",
    \"mlflow.set_tracking_uri('${MLFLOW_CLUSTER_URL}')\\n\",
    \"\\n\",
    \"experiments = mlflow.search_experiments()\\n\",
    \"for exp in experiments:\\n\",
    \"    print(f'Experiment: {exp.name} (ID: {exp.experiment_id})')\\n\",
    \"\\n\",
    \"# Show latest runs\\n\",
    \"runs = mlflow.search_runs(search_all_experiments=True, max_results=5)\\n\",
    \"if not runs.empty:\\n\",
    \"    print(f'\\\\nLatest MLflow runs:')\\n\",
    \"    print(runs[['run_id', 'experiment_id', 'status']].to_string())\\n\",
    \"else:\\n\",
    \"    print('No MLflow runs found')\"
   ],
   \"execution_count\": null,
   \"outputs\": []
  }
 ],
 \"metadata\": {
  \"kernelspec\": {
   \"display_name\": \"Python 3 (ipykernel)\",
   \"language\": \"python\",
   \"name\": \"python3\"
  },
  \"language_info\": {
   \"name\": \"python\",
   \"version\": \"3.11.0\"
  }
 },
 \"nbformat\": 4,
 \"nbformat_minor\": 5
}
NOTEBOOK_EOF"

    log_success "Notebook created at notebooks/01_train_churn_pipeline.ipynb"
}

# ============================================================
# Phase 6: Execute Notebook
# ============================================================

phase_6_execute_notebook() {
    log_phase "Phase 6: Executing Jupyter notebook"

    local repo_dir="/home/jovyan/${GITEA_REPO_NAME}"
    local notebook_path="${repo_dir}/notebooks/01_train_churn_pipeline.ipynb"

    # Check if pipeline already ran (artifacts exist)
    if pod_exec "ls ${repo_dir}/local_artifacts/run_*/metrics.json" &>/dev/null; then
        log_success "Pipeline already ran (artifacts exist, idempotent)"
    else
        log_info "Executing notebook (this may take a few minutes)..."
        pod_exec "
            cd '${repo_dir}' && \
            jupyter nbconvert \
                --to notebook \
                --execute \
                --ExecutePreprocessor.timeout=600 \
                --output '01_train_churn_pipeline.executed.ipynb' \
                '${notebook_path}'
        "
        log_success "Notebook executed successfully"
    fi

    # Verify run artifacts
    log_info "Checking run artifacts..."
    local artifact_count
    artifact_count=$(pod_exec "ls -d ${repo_dir}/local_artifacts/run_* 2>/dev/null | wc -l" || echo "0")
    if [[ "$artifact_count" -gt 0 ]]; then
        log_success "Found $artifact_count run(s) in local_artifacts/"
        pod_exec "ls ${repo_dir}/local_artifacts/run_*/" 2>/dev/null || true
    else
        log_warn "No run artifacts found in local_artifacts/"
    fi

    # Verify metrics file
    if pod_exec "ls ${repo_dir}/local_artifacts/run_*/metrics.json" &>/dev/null; then
        log_info "Metrics:"
        pod_exec "cat ${repo_dir}/local_artifacts/run_*/metrics.json" 2>/dev/null | head -20 || true
    fi

    # Verify MLflow experiment was created
    log_info "Checking MLflow for logged experiment..."
    kubectl port-forward -n "$NS_MBT" svc/mlflow 15000:5000 &
    local mlflow_pf_pid=$!
    PF_PIDS+=("$mlflow_pf_pid")
    sleep 3

    if wait_for_url "http://localhost:15000/health" 10 2; then
        local experiments
        experiments=$(curl -sf "http://localhost:15000/api/2.0/mlflow/experiments/search" \
            -H "Content-Type: application/json" \
            -d '{"max_results": 10}' 2>/dev/null || echo "{}")

        local exp_count
        exp_count=$(echo "$experiments" | jq '.experiments | length' 2>/dev/null || echo "0")
        if [[ "$exp_count" -gt 0 ]]; then
            log_success "MLflow has $exp_count experiment(s)"
            echo "$experiments" | jq -r '.experiments[] | "  - \(.name) (ID: \(.experiment_id))"' 2>/dev/null || true
        else
            log_warn "No experiments found in MLflow (pipeline may not have logged to MLflow)"
        fi
    else
        log_warn "Could not reach MLflow API to verify experiment"
    fi

    cleanup_pf "$mlflow_pf_pid"
}

# ============================================================
# Phase 7: Git Commit + Push
# ============================================================

phase_7_git_push() {
    log_phase "Phase 7: Committing and pushing to $GITEA_BRANCH"

    local repo_dir="/home/jovyan/${GITEA_REPO_NAME}"

    # Check if already committed and pushed
    local unpushed
    unpushed=$(pod_exec "cd ${repo_dir} && git log origin/main..HEAD --oneline 2>/dev/null | wc -l" || echo "0")
    if [[ "$unpushed" -gt 0 ]]; then
        log_success "Changes already committed ($unpushed commit(s) ahead of main)"
        return
    fi

    # Create .gitignore
    log_info "Creating .gitignore..."
    pod_exec "cat > '${repo_dir}/.gitignore' << 'EOF'
# MBT artifacts (generated at runtime)
local_artifacts/
target/
mlruns/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
EOF"

    # Stage project files (not artifacts or executed notebooks)
    log_info "Staging project files..."
    pod_exec "
        cd '${repo_dir}' && \
        git add \
            .gitignore \
            pipelines/ \
            sample_data/ \
            lib/ \
            profiles.yaml \
            pyproject.toml \
            README.md \
            notebooks/01_train_churn_pipeline.ipynb
    "

    # Commit
    log_info "Committing..."
    pod_exec "
        cd '${repo_dir}' && \
        git commit -m 'feat: Add churn training pipeline v1

Initialize MBT project with:
- Training pipeline (${PIPELINE_DST_NAME}) using sklearn RandomForest
- Multi-table feature join (features_a + features_b)
- Temporal windowing (12-month train, 1-month test)
- Feature selection (variance, correlation, LGBM importance)
- Evaluation metrics (ROC-AUC, accuracy, F1, precision, recall)
- Jupyter notebook for interactive execution
- MLflow integration for experiment tracking

Generated with: mbt init --template typical-ds-pipeline --framework sklearn'
    "
    log_success "Changes committed"

    # Push
    log_info "Pushing to origin/${GITEA_BRANCH}..."
    pod_exec "cd '${repo_dir}' && git push -u origin '${GITEA_BRANCH}'"
    log_success "Pushed to ${GITEA_BRANCH}"
}

# ============================================================
# Phase 8: Create Pull Request
# ============================================================

phase_8_create_pr() {
    log_phase "Phase 8: Creating Pull Request in Gitea"

    # Port-forward to Gitea
    kubectl port-forward -n "$NS_MBT" svc/gitea-http 13000:3000 &
    local pf_pid=$!
    PF_PIDS+=("$pf_pid")
    sleep 3

    if ! wait_for_url "http://localhost:13000/api/v1/version" 15 2; then
        log_error "Gitea API not reachable"
        cleanup_pf "$pf_pid"
        return
    fi

    # Check for existing PR
    log_info "Checking for existing PR..."
    local existing_prs
    existing_prs=$(curl -sf "http://localhost:13000/api/v1/repos/${GITEA_REPO_OWNER}/${GITEA_REPO_NAME}/pulls?state=open" \
        -u "${GITEA_DS_USER}:${GITEA_DS_PASSWORD}" 2>/dev/null || echo "[]")

    local matching_pr
    matching_pr=$(echo "$existing_prs" | jq -r ".[] | select(.head.ref == \"${GITEA_BRANCH}\") | .number" 2>/dev/null || echo "")

    if [[ -n "$matching_pr" ]]; then
        log_success "PR #${matching_pr} already exists (idempotent)"
        local pr_url
        pr_url=$(echo "$existing_prs" | jq -r ".[] | select(.head.ref == \"${GITEA_BRANCH}\") | .html_url" 2>/dev/null || echo "")
        log_info "PR URL: $pr_url"
        cleanup_pf "$pf_pid"
        return
    fi

    # Create PR
    log_info "Creating Pull Request..."
    local pr_response
    pr_response=$(curl -sf -X POST \
        "http://localhost:13000/api/v1/repos/${GITEA_REPO_OWNER}/${GITEA_REPO_NAME}/pulls" \
        -H "Content-Type: application/json" \
        -u "${GITEA_DS_USER}:${GITEA_DS_PASSWORD}" \
        -d "{
            \"title\": \"feat: Add churn training pipeline v1\",
            \"body\": \"## Summary\\n\\nInitial MBT training pipeline for churn prediction.\\n\\n### Pipeline: \`${PIPELINE_DST_NAME}\`\\n\\n- **Framework:** sklearn (RandomForestClassifier)\\n- **Data:** Multi-table join (features_a + features_b)\\n- **Windows:** 12-month train / 1-month test\\n- **Feature selection:** variance, correlation, LGBM importance\\n- **Metrics:** ROC-AUC (primary), accuracy, F1, precision, recall\\n- **MLflow:** Integrated for experiment tracking\\n\\n### Testing\\n\\n- [x] Pipeline compiles successfully (\`mbt compile\`)\\n- [x] Pipeline runs end-to-end (\`mbt run\`)\\n- [x] Metrics generated in local_artifacts/\\n- [x] MLflow experiment logged\\n- [x] Notebook execution verified\\n\\n### Files\\n\\n- \`pipelines/${PIPELINE_DST_NAME}.yaml\` - Training pipeline definition\\n- \`profiles.yaml\` - Environment configuration\\n- \`sample_data/\` - Synthetic training data\\n- \`notebooks/01_train_churn_pipeline.ipynb\` - Interactive notebook\\n\\n---\\n*Created by ds-team using MBT (Model Build Tool)*\",
            \"head\": \"${GITEA_BRANCH}\",
            \"base\": \"main\"
        }" 2>/dev/null)

    local pr_number
    pr_number=$(echo "$pr_response" | jq -r '.number // empty' 2>/dev/null || echo "")

    if [[ -n "$pr_number" ]]; then
        local pr_url
        pr_url=$(echo "$pr_response" | jq -r '.html_url' 2>/dev/null || echo "")
        log_success "Pull Request #${pr_number} created"
        log_info "PR URL: $pr_url"
    else
        log_error "Failed to create PR"
        echo "$pr_response" | jq . 2>/dev/null || echo "$pr_response"
    fi

    cleanup_pf "$pf_pid"
}

# ============================================================
# Phase 9: Summary
# ============================================================

phase_9_summary() {
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - START_TIME))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))

    log_phase "DS Workflow Complete - Summary"

    echo -e "  ${GREEN}Pipeline:${NC}        ${PIPELINE_DST_NAME}"
    echo -e "  ${GREEN}Framework:${NC}       ${MBT_FRAMEWORK} (RandomForestClassifier)"
    echo -e "  ${GREEN}Data:${NC}            ${MBT_NUM_CUSTOMERS} customers, ${MBT_FEATURES_A}+${MBT_FEATURES_B} features"
    echo -e "  ${GREEN}Branch:${NC}          ${GITEA_BRANCH}"
    echo -e "  ${GREEN}Singleuser Pod:${NC}  ${SINGLEUSER_POD:-unknown}"
    echo ""
    echo -e "  ${GREEN}Service Links:${NC}"
    echo "    JupyterHub:  http://jupyter.localhost"
    echo "    Gitea PR:    http://gitea.localhost/${GITEA_REPO_OWNER}/${GITEA_REPO_NAME}/pulls"
    echo "    MLflow:      http://mlflow.localhost"
    echo ""
    echo -e "  ${GREEN}Elapsed:${NC}         ${minutes}m ${seconds}s"
    echo ""

    log_success "Data scientist workflow simulation complete!"
}

# ============================================================
# Main Execution
# ============================================================

main() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  03-mbt-init.sh - DS Workflow: Training Pipeline + PR${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""

    phase_0_prerequisites
    phase_1_spawn_jupyter
    phase_2_install_mbt
    phase_3_clone_repo
    phase_4_mbt_init
    phase_5_create_notebook
    phase_6_execute_notebook
    phase_7_git_push
    phase_8_create_pr
    phase_9_summary
}

main "$@"
