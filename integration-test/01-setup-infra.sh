#!/usr/bin/env bash
# 01-setup-infra.sh - Deploy MBT integration test infrastructure
#
# Supports two Kubernetes providers:
#   - k3d:              Creates a dedicated k3d cluster (default if k3d is installed)
#   - rancher-desktop:  Uses the existing Rancher Desktop k3s cluster
#
# Auto-detects the provider, or set MBT_K8S_PROVIDER=k3d|rancher-desktop to override.
#
# Creates a Kubernetes environment with all services needed for the MBT
# integration test: PostgreSQL, SeaweedFS, MLflow, Gitea, Woodpecker CI,
# Zot Registry, Airflow, H2O, JupyterHub, Metabase, Prometheus, Grafana.
#
# Usage:
#   ./01-setup-infra.sh           # Full setup (auto-detect provider)
#   ./01-setup-infra.sh teardown  # Destroy cluster / clean up resources
#   ./01-setup-infra.sh status    # Health check all services
#
# Prerequisites:
#   k3d:              k3d (>=v5.3.0), helm, kubectl, docker, curl, jq
#   rancher-desktop:  rdctl, helm, kubectl, docker, curl, jq
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR/infra/k8s"
MBT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# Provider Detection
# ============================================================

# Supported providers: k3d, rancher-desktop
# Override with MBT_K8S_PROVIDER environment variable
detect_provider() {
    if [[ -n "${MBT_K8S_PROVIDER:-}" ]]; then
        case "$MBT_K8S_PROVIDER" in
            k3d|rancher-desktop) echo "$MBT_K8S_PROVIDER" ;;
            *) echo ""; return 1 ;;
        esac
        return
    fi

    # Check current kubectl context first
    local current_ctx
    current_ctx=$(kubectl config current-context 2>/dev/null || echo "")
    if [[ "$current_ctx" == "rancher-desktop" ]]; then
        echo "rancher-desktop"
        return
    fi

    # Fall back to binary detection (prefer k3d if both exist)
    if command -v k3d &>/dev/null; then
        echo "k3d"
    elif command -v rdctl &>/dev/null; then
        echo "rancher-desktop"
    else
        echo ""
    fi
}

K8S_PROVIDER=$(detect_provider)

if [[ -z "$K8S_PROVIDER" ]]; then
    echo "ERROR: Could not detect Kubernetes provider."
    echo "Install k3d or Rancher Desktop, or set MBT_K8S_PROVIDER=k3d|rancher-desktop"
    exit 1
fi

# Helper: get the kubectl context name for the current provider
kube_context() {
    case "$K8S_PROVIDER" in
        k3d) echo "k3d-${CLUSTER_NAME}" ;;
        rancher-desktop) echo "rancher-desktop" ;;
    esac
}

# ============================================================
# Configuration
# ============================================================

CLUSTER_NAME="mbt-integration"

# Namespaces
NS_MBT="mbt"
NS_PIPELINES="mbt-pipelines"
NS_MONITORING="mbt-monitoring"

# Credentials - PostgreSQL
PG_ADMIN_PASSWORD="admin_password"
PG_MLFLOW_PASSWORD="mlflow_password"
PG_AIRFLOW_PASSWORD="airflow_password"
PG_MBT_PASSWORD="mbt_password"
PG_METABASE_PASSWORD="metabase_password"
PG_GITEA_PASSWORD="gitea_password"

# Credentials - S3 (SeaweedFS)
S3_ACCESS_KEY="mbt-access-key"
S3_SECRET_KEY="mbt-secret-key"

# Credentials - Gitea
GITEA_ADMIN_USER="mbt-admin"
GITEA_ADMIN_PASSWORD="mbt-admin-password"
GITEA_ADMIN_EMAIL="admin@mbt.local"

# Credentials - Airflow
AIRFLOW_ADMIN_USER="admin"
AIRFLOW_ADMIN_PASSWORD="admin"

# Credentials - Grafana
GRAFANA_ADMIN_PASSWORD="mbt-grafana"

# Credentials - JupyterHub
JUPYTERHUB_PASSWORD="mbt-jupyter"

# Zot registry port (NodePort)
ZOT_NODEPORT=30050
ZOT_HOST="localhost:${ZOT_NODEPORT}"

# ============================================================
# Utility Functions
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

wait_for_pod() {
    local namespace="$1" label="$2" timeout="${3:-300}"
    log_info "Waiting for pod $label in $namespace (timeout: ${timeout}s)..."
    if ! kubectl wait --for=condition=ready pod -l "$label" \
        -n "$namespace" --timeout="${timeout}s" 2>/dev/null; then
        log_warn "Pod $label not ready within ${timeout}s, continuing..."
        return 1
    fi
    return 0
}

wait_for_job() {
    local namespace="$1" job_name="$2" timeout="${3:-120}"
    log_info "Waiting for job $job_name in $namespace..."
    kubectl wait --for=condition=complete "job/$job_name" \
        -n "$namespace" --timeout="${timeout}s" 2>/dev/null || true
}

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

helm_install() {
    local release="$1" chart="$2" namespace="$3"
    shift 3
    helm upgrade --install "$release" "$chart" \
        --namespace "$namespace" --create-namespace \
        --wait --timeout 300s \
        "$@"
}

# Kill background port-forward process cleanly
cleanup_pf() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

# ============================================================
# Phase 0: Prerequisites
# ============================================================

phase_0_prerequisites() {
    log_phase "Phase 0: Checking prerequisites (provider: $K8S_PROVIDER)"

    # Common tools required by all providers
    local required_cmds=(kubectl helm docker curl jq)

    # Provider-specific tools
    case "$K8S_PROVIDER" in
        k3d)
            required_cmds+=(k3d)
            ;;
        rancher-desktop)
            required_cmds+=(rdctl)
            ;;
    esac

    local missing=()
    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Install them before running this script."
        exit 1
    fi

    # Provider-specific version/status checks
    case "$K8S_PROVIDER" in
        k3d)
            local k3d_ver
            k3d_ver=$(k3d version 2>/dev/null | head -1 | grep -oP 'v\K[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
            log_info "k3d version: $k3d_ver"
            ;;
        rancher-desktop)
            local rd_ver
            rd_ver=$(rdctl version 2>/dev/null | jq -r '.version // "unknown"' 2>/dev/null || echo "unknown")
            log_info "Rancher Desktop version: $rd_ver"

            # Verify Rancher Desktop is running
            if ! rdctl list-settings &>/dev/null; then
                log_error "Rancher Desktop is not running. Start it before running this script."
                exit 1
            fi
            ;;
    esac

    # Check Docker daemon
    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Add required Helm repos (idempotent)
    log_info "Adding Helm repositories..."
    helm repo add bitnami https://charts.bitnami.com/bitnami 2>/dev/null || true
    helm repo add seaweedfs https://seaweedfs.github.io/seaweedfs/helm 2>/dev/null || true
    helm repo add gitea https://dl.gitea.io/charts/ 2>/dev/null || true
    # Woodpecker uses OCI registry, no helm repo add needed
    helm repo add apache-airflow https://airflow.apache.org 2>/dev/null || true
    helm repo add jupyterhub https://hub.jupyter.org/helm-chart/ 2>/dev/null || true
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
    helm repo add traefik https://traefik.github.io/charts 2>/dev/null || true
    helm repo add zotregistry https://zotregistry.dev/helm-charts/ 2>/dev/null || true
    helm repo update || log_warn "Some Helm repos failed to update (may still work with cached index)"

    log_success "All prerequisites satisfied"
}

# ============================================================
# Phase 1: Kubernetes Cluster
# ============================================================

phase_1_cluster() {
    log_phase "Phase 1: Setting up Kubernetes cluster ($K8S_PROVIDER)"

    case "$K8S_PROVIDER" in
        k3d)
            _phase_1_k3d
            ;;
        rancher-desktop)
            _phase_1_rancher_desktop
            ;;
    esac
}

_phase_1_k3d() {
    # Check if cluster already exists
    if k3d cluster list -o json 2>/dev/null | jq -e ".[] | select(.name==\"$CLUSTER_NAME\")" &>/dev/null; then
        log_warn "Cluster '$CLUSTER_NAME' already exists, skipping creation"
        k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-switch-context 2>/dev/null
        return 0
    fi

    # Create registries.yaml for containerd to pull from Zot inside the cluster
    local registries_config
    registries_config=$(mktemp /tmp/mbt-k3d-registries.XXXXXX.yaml)
    cat > "$registries_config" <<'EOF'
mirrors:
  "zot-registry.mbt.svc.cluster.local:5000":
    endpoint:
      - "http://zot-registry.mbt.svc.cluster.local:5000"
configs:
  "zot-registry.mbt.svc.cluster.local:5000":
    tls:
      insecure_skip_verify: true
EOF

    log_info "Creating k3d cluster '$CLUSTER_NAME'..."
    k3d cluster create "$CLUSTER_NAME" \
        --servers 1 \
        --agents 2 \
        --api-port 6550 \
        --port "80:80@loadbalancer" \
        --port "443:443@loadbalancer" \
        --port "30000-30500:30000-30500@server:0" \
        --k3s-arg "--disable=traefik@server:0" \
        --k3s-arg "--kubelet-arg=max-pods=200@agent:*" \
        --registry-config "$registries_config" \
        --wait

    rm -f "$registries_config"

    # Merge kubeconfig
    k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-switch-context

    # Wait for nodes
    log_info "Waiting for cluster nodes to be ready..."
    kubectl wait --for=condition=ready node --all --timeout=120s

    # Install Traefik via Helm (instead of k3s built-in)
    log_info "Installing Traefik ingress controller..."
    helm_install traefik traefik/traefik kube-system \
        --set "ports.web.nodePort=30080" \
        --set "ports.websecure.nodePort=30443" \
        --set "service.type=LoadBalancer" \
        --set "ingressRoute.dashboard.enabled=false"

    log_success "k3d cluster '$CLUSTER_NAME' is ready"
}

_phase_1_rancher_desktop() {
    # Switch to rancher-desktop context
    log_info "Switching to rancher-desktop kubectl context..."
    kubectl config use-context rancher-desktop 2>/dev/null \
        || { log_error "kubectl context 'rancher-desktop' not found. Is Rancher Desktop running with Kubernetes enabled?"; exit 1; }

    # Verify cluster is reachable
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot reach Rancher Desktop Kubernetes cluster. Ensure Kubernetes is enabled in Rancher Desktop settings."
        exit 1
    fi

    # Wait for nodes
    log_info "Waiting for cluster nodes to be ready..."
    kubectl wait --for=condition=ready node --all --timeout=120s

    # Configure containerd registry mirror for Zot (insecure localhost registry)
    # Rancher Desktop's k3s reads registries.yaml from /etc/rancher/k3s/
    log_info "Configuring containerd registry mirror for Zot..."
    local registries_yaml="/etc/rancher/k3s/registries.yaml"
    local registries_content
    registries_content=$(cat <<'REGEOF'
mirrors:
  "zot-registry.mbt.svc.cluster.local:5000":
    endpoint:
      - "http://zot-registry.mbt.svc.cluster.local:5000"
configs:
  "zot-registry.mbt.svc.cluster.local:5000":
    tls:
      insecure_skip_verify: true
REGEOF
)

    if rdctl shell test -f "$registries_yaml" 2>/dev/null; then
        log_warn "Registries config already exists at $registries_yaml, skipping"
    else
        echo "$registries_content" | rdctl shell sudo tee "$registries_yaml" > /dev/null 2>&1 \
            || log_warn "Could not write registries.yaml via rdctl. Images may need manual containerd config."
    fi

    # Rancher Desktop ships with Traefik built-in — no need to install via Helm
    log_info "Using Rancher Desktop's built-in Traefik ingress controller"

    log_success "Rancher Desktop cluster is ready"
}

# ============================================================
# Phase 2: Namespaces + Shared Resources
# ============================================================

phase_2_shared_resources() {
    log_phase "Phase 2: Creating namespaces and shared resources"

    kubectl apply -f "$K8S_DIR/namespaces.yaml"

    # PostgreSQL credentials secret (used by MLflow, Airflow, Metabase, MBT)
    kubectl create secret generic postgres-credentials \
        --namespace "$NS_MBT" \
        --from-literal=admin-password="$PG_ADMIN_PASSWORD" \
        --from-literal=mlflow-password="$PG_MLFLOW_PASSWORD" \
        --from-literal=airflow-password="$PG_AIRFLOW_PASSWORD" \
        --from-literal=mbt-password="$PG_MBT_PASSWORD" \
        --from-literal=metabase-password="$PG_METABASE_PASSWORD" \
        --from-literal=gitea-password="$PG_GITEA_PASSWORD" \
        --dry-run=client -o yaml | kubectl apply -f -

    # S3 credentials secret
    kubectl create secret generic s3-credentials \
        --namespace "$NS_MBT" \
        --from-literal=access-key="$S3_ACCESS_KEY" \
        --from-literal=secret-key="$S3_SECRET_KEY" \
        --dry-run=client -o yaml | kubectl apply -f -

    # SeaweedFS S3 identity config (required for S3 API auth)
    local s3_config
    s3_config=$(cat <<EOFS3
{
  "identities": [
    {
      "name": "mbt-admin",
      "credentials": [
        {
          "accessKey": "${S3_ACCESS_KEY}",
          "secretKey": "${S3_SECRET_KEY}"
        }
      ],
      "actions": ["Admin", "Read", "Write", "List", "Tagging"]
    }
  ]
}
EOFS3
)
    kubectl create secret generic seaweedfs-s3-config \
        --namespace "$NS_MBT" \
        --from-literal=seaweedfs_s3_config="$s3_config" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Copy secrets to mbt-pipelines namespace for pipeline pods
    kubectl create secret generic s3-credentials \
        --namespace "$NS_PIPELINES" \
        --from-literal=access-key="$S3_ACCESS_KEY" \
        --from-literal=secret-key="$S3_SECRET_KEY" \
        --dry-run=client -o yaml | kubectl apply -f -

    kubectl create secret generic postgres-credentials \
        --namespace "$NS_PIPELINES" \
        --from-literal=mbt-password="$PG_MBT_PASSWORD" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Shared ConfigMap with service endpoints (for in-cluster DNS)
    kubectl create configmap mbt-env-config \
        --namespace "$NS_MBT" \
        --from-literal=POSTGRES_HOST="postgres-postgresql.mbt.svc.cluster.local" \
        --from-literal=POSTGRES_PORT="5432" \
        --from-literal=SEAWEEDFS_S3_ENDPOINT="http://seaweedfs-filer.mbt.svc.cluster.local:8333" \
        --from-literal=MLFLOW_TRACKING_URI="http://mlflow.mbt.svc.cluster.local:5000" \
        --from-literal=H2O_URL="http://h2o.mbt.svc.cluster.local:54321" \
        --from-literal=GITEA_URL="http://gitea-http.mbt.svc.cluster.local:3000" \
        --from-literal=AIRFLOW_URL="http://airflow-api-server.mbt.svc.cluster.local:8080" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Create postgres init.sql as ConfigMap
    kubectl create configmap postgres-init-sql \
        --namespace "$NS_MBT" \
        --from-file=init.sql="$SCRIPT_DIR/infra/postgres/init.sql" \
        --dry-run=client -o yaml | kubectl apply -f -

    log_success "Shared resources created"
}

# ============================================================
# Phase 3: PostgreSQL
# ============================================================

phase_3_postgres() {
    log_phase "Phase 3: Deploying PostgreSQL"

    helm_install postgres bitnami/postgresql "$NS_MBT" \
        --version "18.2.0" \
        -f "$K8S_DIR/postgres/values.yaml"

    wait_for_pod "$NS_MBT" "app.kubernetes.io/name=postgresql" 180

    log_success "PostgreSQL deployed (NodePort: localhost:30432)"
}

# ============================================================
# Phase 4: SeaweedFS
# ============================================================

phase_4_seaweedfs() {
    log_phase "Phase 4: Deploying SeaweedFS"

    helm_install seaweedfs seaweedfs/seaweedfs "$NS_MBT" \
        -f "$K8S_DIR/seaweedfs/values.yaml"

    wait_for_pod "$NS_MBT" "app.kubernetes.io/name=seaweedfs,app.kubernetes.io/component=filer" 180

    # Create NodePort service for S3 API access from host
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: seaweedfs-s3-nodeport
  namespace: $NS_MBT
spec:
  type: NodePort
  selector:
    app.kubernetes.io/name: seaweedfs
    app.kubernetes.io/component: filer
  ports:
    - name: s3
      port: 8333
      targetPort: 8333
      nodePort: 30333
EOF

    # Buckets are created by the Helm chart via createBuckets in values.yaml

    log_success "SeaweedFS deployed with S3 buckets (NodePort: localhost:30333)"
}

# ============================================================
# Phase 5: Zot Registry
# ============================================================

phase_5_zot_registry() {
    log_phase "Phase 5: Deploying Zot container registry"

    helm_install zot-registry zotregistry/zot "$NS_MBT" \
        -f "$K8S_DIR/zot/values.yaml"

    wait_for_pod "$NS_MBT" "app.kubernetes.io/name=zot" 120

    # Wait for Zot to be accessible from host
    log_info "Waiting for Zot registry at localhost:${ZOT_NODEPORT}..."
    if wait_for_url "http://localhost:${ZOT_NODEPORT}/v2/" 30 2; then
        log_success "Zot registry accessible at localhost:${ZOT_NODEPORT}"
    else
        log_warn "Zot registry not yet accessible from host, may need time"
    fi

    log_success "Zot registry deployed (NodePort: localhost:${ZOT_NODEPORT})"
}

# ============================================================
# Phase 6: Build and Push Custom Images
# ============================================================

phase_6_build_images() {
    log_phase "Phase 6: Building and pushing custom images"

    # Build MLflow image (v3.10.0)
    log_info "Building MLflow image..."
    docker build -t "${ZOT_HOST}/mbt/mlflow:3.10.0" \
        -f "$SCRIPT_DIR/infra/mlflow/Dockerfile" \
        "$SCRIPT_DIR/infra/mlflow/"
    docker push "${ZOT_HOST}/mbt/mlflow:3.10.0"
    log_success "MLflow image pushed to Zot"

    # Build Airflow image (v3.1.7)
    log_info "Building Airflow image..."
    docker build -t "${ZOT_HOST}/mbt/airflow:3.1.7" \
        -f "$SCRIPT_DIR/infra/airflow/Dockerfile" \
        "$SCRIPT_DIR/infra/airflow/"
    docker push "${ZOT_HOST}/mbt/airflow:3.1.7"
    log_success "Airflow image pushed to Zot"

    # Build MBT runner image (context is repo root)
    log_info "Building MBT runner image..."
    docker build -t "${ZOT_HOST}/mbt/mbt-runner:latest" \
        -f "$SCRIPT_DIR/infra/mbt-runner/Dockerfile" \
        "$MBT_ROOT"
    docker push "${ZOT_HOST}/mbt/mbt-runner:latest"
    log_success "MBT runner image pushed to Zot"

    # Import images into cluster nodes (fallback in case containerd mirror is not yet configured)
    local images=(
        "${ZOT_HOST}/mbt/mlflow:3.10.0"
        "${ZOT_HOST}/mbt/airflow:3.1.7"
        "${ZOT_HOST}/mbt/mbt-runner:latest"
    )

    case "$K8S_PROVIDER" in
        k3d)
            log_info "Importing images into k3d cluster..."
            k3d image import "${images[@]}" \
                -c "$CLUSTER_NAME" 2>/dev/null \
                || log_warn "k3d image import had issues, images may still work via Zot"
            ;;
        rancher-desktop)
            log_info "Importing images into Rancher Desktop's k3s containerd..."
            local import_ok=true
            for img in "${images[@]}"; do
                docker save "$img" | rdctl shell sudo ctr --address /run/k3s/containerd/containerd.sock \
                    --namespace k8s.io images import - 2>/dev/null \
                    || { import_ok=false; break; }
            done
            if ! $import_ok; then
                log_warn "Image import via rdctl failed. Trying nerdctl..."
                for img in "${images[@]}"; do
                    docker save "$img" | nerdctl --namespace k8s.io load 2>/dev/null || true
                done
                log_warn "Images may still work via Zot registry pull"
            fi
            ;;
    esac

    log_success "All custom images built and pushed"
}

# ============================================================
# Phase 7: MLflow
# ============================================================

phase_7_mlflow() {
    log_phase "Phase 7: Deploying MLflow"

    kubectl apply -f "$K8S_DIR/mlflow/deployment.yaml"
    kubectl apply -f "$K8S_DIR/mlflow/service.yaml"
    kubectl apply -f "$K8S_DIR/mlflow/ingress.yaml"

    wait_for_pod "$NS_MBT" "app.kubernetes.io/name=mlflow" 180 || true

    log_success "MLflow deployed at http://mlflow.localhost"
}

# ============================================================
# Phase 8: Gitea
# ============================================================

phase_8_gitea() {
    log_phase "Phase 8: Deploying Gitea"

    helm_install gitea gitea/gitea "$NS_MBT" \
        -f "$K8S_DIR/gitea/values.yaml"

    wait_for_pod "$NS_MBT" "app.kubernetes.io/name=gitea" 180 || true

    # Wait for Gitea API to be ready
    log_info "Waiting for Gitea API to be available..."
    kubectl port-forward -n "$NS_MBT" svc/gitea-http 3000:3000 &
    local pf_pid=$!
    sleep 5

    if wait_for_url "http://localhost:3000/api/v1/version" 30 3; then
        log_success "Gitea API is ready"

        # Create DS user
        log_info "Creating ds-team user..."
        curl -sf -X POST "http://localhost:3000/api/v1/admin/users" \
            -H "Content-Type: application/json" \
            -u "${GITEA_ADMIN_USER}:${GITEA_ADMIN_PASSWORD}" \
            -d '{
                "username": "ds-team",
                "password": "ds-team-password",
                "email": "ds-team@mbt.local",
                "must_change_password": false,
                "login_name": "ds-team",
                "source_id": 0
            }' > /dev/null 2>&1 || log_warn "ds-team user may already exist"

        # Create DE user
        log_info "Creating de-team user..."
        curl -sf -X POST "http://localhost:3000/api/v1/admin/users" \
            -H "Content-Type: application/json" \
            -u "${GITEA_ADMIN_USER}:${GITEA_ADMIN_PASSWORD}" \
            -d '{
                "username": "de-team",
                "password": "de-team-password",
                "email": "de-team@mbt.local",
                "must_change_password": false,
                "login_name": "de-team",
                "source_id": 0
            }' > /dev/null 2>&1 || log_warn "de-team user may already exist"

    else
        log_warn "Gitea API not ready, manual user creation may be needed"
    fi

    cleanup_pf "$pf_pid"

    log_success "Gitea deployed at http://gitea.localhost"
}

# ============================================================
# Phase 9: Woodpecker CI
# ============================================================

phase_9_woodpecker() {
    log_phase "Phase 9: Deploying Woodpecker CI"

    # Create OAuth2 application in Gitea for Woodpecker
    log_info "Creating Gitea OAuth2 application for Woodpecker..."
    kubectl port-forward -n "$NS_MBT" svc/gitea-http 3000:3000 &
    local pf_pid=$!
    sleep 3

    local oauth_response
    oauth_response=$(curl -sf -X POST \
        "http://localhost:3000/api/v1/user/applications/oauth2" \
        -H "Content-Type: application/json" \
        -u "${GITEA_ADMIN_USER}:${GITEA_ADMIN_PASSWORD}" \
        -d '{
            "name": "woodpecker-ci",
            "redirect_uris": ["http://ci.localhost/authorize"],
            "confidential_client": true
        }' 2>/dev/null || echo '{}')

    local gitea_client_id gitea_client_secret
    gitea_client_id=$(echo "$oauth_response" | jq -r '.client_id // empty')
    gitea_client_secret=$(echo "$oauth_response" | jq -r '.client_secret // empty')

    if [[ -z "$gitea_client_id" || -z "$gitea_client_secret" ]]; then
        log_warn "OAuth2 app may already exist, trying to find it..."
        local existing
        existing=$(curl -sf "http://localhost:3000/api/v1/user/applications/oauth2" \
            -u "${GITEA_ADMIN_USER}:${GITEA_ADMIN_PASSWORD}" 2>/dev/null || echo '[]')
        gitea_client_id=$(echo "$existing" | jq -r '.[] | select(.name=="woodpecker-ci") | .client_id // empty')

        if [[ -z "$gitea_client_id" ]]; then
            log_error "Failed to create or find Gitea OAuth2 application"
            cleanup_pf "$pf_pid"
            # Create a placeholder secret so Woodpecker can still start
            gitea_client_id="placeholder-client-id"
            gitea_client_secret="placeholder-client-secret"
            log_warn "Using placeholder OAuth2 credentials. Configure manually at http://gitea.localhost"
        else
            gitea_client_secret="(existing - check Gitea UI or re-create)"
            log_warn "Found existing OAuth2 app. Secret not retrievable. Delete and re-create if needed."
        fi
    else
        log_success "OAuth2 app created: client_id=$gitea_client_id"
    fi

    cleanup_pf "$pf_pid"

    # Create K8s secret with OAuth2 credentials
    kubectl create secret generic woodpecker-gitea-oauth \
        --namespace "$NS_MBT" \
        --from-literal=WOODPECKER_GITEA_CLIENT="$gitea_client_id" \
        --from-literal=WOODPECKER_GITEA_SECRET="$gitea_client_secret" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Deploy Woodpecker (OCI registry — not a traditional helm repo)
    helm upgrade --install woodpecker oci://ghcr.io/woodpecker-ci/helm/woodpecker \
        --namespace "$NS_MBT" \
        -f "$K8S_DIR/woodpecker/values.yaml" \
        --wait --timeout 300s || log_warn "Woodpecker helm install had issues"

    wait_for_pod "$NS_MBT" "app.kubernetes.io/name=woodpecker,app.kubernetes.io/component=server" 120 || true

    log_success "Woodpecker CI deployed at http://ci.localhost"
}

# ============================================================
# Phase 10: Airflow
# ============================================================

phase_10_airflow() {
    log_phase "Phase 10: Deploying Airflow"

    # Create Airflow database connection secret
    kubectl create secret generic airflow-db-secret \
        --namespace "$NS_MBT" \
        --from-literal=connection="postgresql://airflow_user:${PG_AIRFLOW_PASSWORD}@postgres-postgresql.mbt.svc.cluster.local:5432/airflow_db" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Create Fernet key secret
    local fernet_key
    fernet_key=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null \
        || echo "ZmRfd2ViaGVhZF9mZXJuZXRfa2V5X2Zvcl90ZXN0aW5n")

    kubectl create secret generic airflow-fernet-key \
        --namespace "$NS_MBT" \
        --from-literal=fernet-key="$fernet_key" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Deploy Airflow — do NOT use --wait here. The chart uses post-install Helm hooks
    # (db migrate + create user) that need to complete before pods become ready.
    # Using --wait would cause a deadlock: Helm waits for pods, pods wait for migration.
    helm upgrade --install airflow apache-airflow/airflow \
        --namespace "$NS_MBT" --create-namespace \
        -f "$K8S_DIR/airflow/values.yaml" \
        --set "fernetKeySecretName=airflow-fernet-key" \
        --timeout 600s || log_warn "Airflow helm install had issues (may need more time)"

    # Create RBAC for Airflow to manage pods in mbt-pipelines namespace
    kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: airflow-pod-operator
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "pods/status"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["events"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: airflow-pod-operator-binding
subjects:
  - kind: ServiceAccount
    name: airflow
    namespace: $NS_MBT
roleRef:
  kind: ClusterRole
  name: airflow-pod-operator
  apiGroup: rbac.authorization.k8s.io
EOF

    wait_for_pod "$NS_MBT" "component=api-server" 300 || true

    log_success "Airflow deployed at http://airflow.localhost"
}

# ============================================================
# Phase 11: H2O Server
# ============================================================

phase_11_h2o() {
    log_phase "Phase 11: Deploying H2O Server"

    kubectl apply -f "$K8S_DIR/h2o/deployment.yaml"
    kubectl apply -f "$K8S_DIR/h2o/service.yaml"

    wait_for_pod "$NS_MBT" "app.kubernetes.io/name=h2o" 180 || true

    log_success "H2O server deployed (cluster: http://h2o:54321, host: localhost:30054)"
}

# ============================================================
# Phase 12: JupyterHub
# ============================================================

phase_12_jupyterhub() {
    log_phase "Phase 12: Deploying JupyterHub"

    helm_install jupyterhub jupyterhub/jupyterhub "$NS_MBT" \
        -f "$K8S_DIR/jupyterhub/values.yaml" \
        --timeout 300s || log_warn "JupyterHub helm install had issues"

    wait_for_pod "$NS_MBT" "app=jupyterhub,component=hub" 180 || true

    log_success "JupyterHub deployed at http://jupyter.localhost"
}

# ============================================================
# Phase 13: Metabase
# ============================================================

phase_13_metabase() {
    log_phase "Phase 13: Deploying Metabase"

    kubectl apply -f "$K8S_DIR/metabase/deployment.yaml"
    kubectl apply -f "$K8S_DIR/metabase/service.yaml"
    kubectl apply -f "$K8S_DIR/metabase/ingress.yaml"

    # Metabase takes a long time to start (Java app, DB migration)
    wait_for_pod "$NS_MBT" "app.kubernetes.io/name=metabase" 300 || true

    log_success "Metabase deployed at http://metabase.localhost"
}

# ============================================================
# Phase 14: Monitoring (Prometheus + Grafana)
# ============================================================

phase_14_monitoring() {
    log_phase "Phase 14: Deploying Prometheus + Grafana"

    helm_install prometheus prometheus-community/kube-prometheus-stack "$NS_MONITORING" \
        -f "$K8S_DIR/monitoring/prometheus-values.yaml" \
        --timeout 300s || log_warn "Monitoring helm install had issues"

    wait_for_pod "$NS_MONITORING" "app.kubernetes.io/name=grafana" 180 || true

    log_success "Monitoring deployed: Grafana at http://grafana.localhost"
}

# ============================================================
# Phase 15: MBT Pipeline Namespace RBAC + ConfigMap
# ============================================================

phase_15_mbt_pipelines_rbac() {
    log_phase "Phase 15: Configuring MBT pipeline namespace"

    kubectl apply -f "$K8S_DIR/mbt-pipelines/rbac.yaml"
    kubectl apply -f "$K8S_DIR/mbt-pipelines/configmap.yaml"

    log_success "MBT pipeline namespace configured (ns: $NS_PIPELINES, sa: mbt-runner)"
}

# ============================================================
# Phase 16: Post-deployment Initialization
# ============================================================

phase_16_post_init() {
    log_phase "Phase 16: Post-deployment initialization"

    # 16a: Verify PostgreSQL data
    log_info "Verifying PostgreSQL data..."
    local row_count
    row_count=$(kubectl exec -n "$NS_MBT" deploy/postgres-postgresql -- \
        psql -U postgres -d warehouse -t -c "SELECT COUNT(*) FROM customers;" 2>/dev/null | tr -d ' ' || echo "0")
    if [[ "$row_count" -gt 0 ]] 2>/dev/null; then
        log_success "PostgreSQL warehouse has $row_count customers"
    else
        log_warn "No customer data found; init.sql may not have run correctly"
    fi

    # 16b: Create MBT project repo in Gitea
    log_info "Creating MBT project repository in Gitea..."
    kubectl port-forward -n "$NS_MBT" svc/gitea-http 3000:3000 &
    local pf_pid=$!
    sleep 3

    # Create repo under de-team user
    curl -sf -X POST "http://localhost:3000/api/v1/user/repos" \
        -H "Content-Type: application/json" \
        -u "de-team:de-team-password" \
        -d '{
            "name": "ml-pipeline",
            "description": "MBT integration test ML pipeline project",
            "auto_init": true,
            "default_branch": "main",
            "private": false
        }' > /dev/null 2>&1 || log_warn "Repo ml-pipeline may already exist"

    # Add ds-team as collaborator
    curl -sf -X PUT "http://localhost:3000/api/v1/repos/de-team/ml-pipeline/collaborators/ds-team" \
        -H "Content-Type: application/json" \
        -u "de-team:de-team-password" \
        -d '{"permission": "write"}' > /dev/null 2>&1 || log_warn "Collaborator may already be added"

    cleanup_pf "$pf_pid"

    # 16c: Verify S3 buckets
    log_info "Verifying S3 buckets..."
    local bucket_list
    bucket_list=$(AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" \
        aws --endpoint-url "http://localhost:30333" s3 ls 2>/dev/null || echo "")
    if echo "$bucket_list" | grep -q "mbt-pipeline-artifacts"; then
        log_success "S3 buckets verified"
    else
        log_warn "S3 buckets may not be ready yet"
    fi

    log_success "Post-deployment initialization complete"
}

# ============================================================
# Phase 17: Health Check + Summary
# ============================================================

phase_17_summary() {
    log_phase "Phase 17: Health check and service summary"

    local all_ok=true

    # Check custom deployments
    for deploy in mlflow h2o metabase; do
        if kubectl rollout status "deployment/$deploy" -n "$NS_MBT" --timeout=30s &>/dev/null; then
            log_success "  $deploy: ready"
        else
            log_warn "  $deploy: not ready yet"
            all_ok=false
        fi
    done

    # Check Helm releases in mbt namespace
    for release in postgres seaweedfs gitea airflow jupyterhub zot-registry woodpecker; do
        local status
        status=$(helm status "$release" -n "$NS_MBT" -o json 2>/dev/null | jq -r '.info.status' || echo "not found")
        if [[ "$status" == "deployed" ]]; then
            log_success "  $release (helm): deployed"
        else
            log_warn "  $release (helm): $status"
            all_ok=false
        fi
    done

    # Check monitoring
    local mon_status
    mon_status=$(helm status prometheus -n "$NS_MONITORING" -o json 2>/dev/null | jq -r '.info.status' || echo "not found")
    if [[ "$mon_status" == "deployed" ]]; then
        log_success "  prometheus+grafana (helm): deployed"
    else
        log_warn "  prometheus+grafana (helm): $mon_status"
        all_ok=false
    fi

    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  MBT Integration Test Infrastructure - Service Access${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    echo -e "  ${GREEN}HTTP Services (via Traefik Ingress):${NC}"
    echo "    MLflow:       http://mlflow.localhost"
    echo "    Airflow:      http://airflow.localhost"
    echo "    Gitea:        http://gitea.localhost"
    echo "    Woodpecker:   http://ci.localhost"
    echo "    JupyterHub:   http://jupyter.localhost"
    echo "    Metabase:     http://metabase.localhost"
    echo "    Grafana:      http://grafana.localhost"
    echo "    Prometheus:   http://prometheus.localhost"
    echo ""
    echo -e "  ${GREEN}Non-HTTP Services (NodePort):${NC}"
    echo "    PostgreSQL:   localhost:30432"
    echo "    SeaweedFS S3: localhost:30333"
    echo "    H2O Server:   localhost:30054"
    echo "    Zot Registry: localhost:${ZOT_NODEPORT}"
    echo "    Gitea SSH:    localhost:30022"
    echo ""
    echo -e "  ${GREEN}Credentials:${NC}"
    echo ""
    echo "    Service          Username             Password / Secret"
    echo "    -------          --------             -----------------"
    echo "    Airflow          ${AIRFLOW_ADMIN_USER}                ${AIRFLOW_ADMIN_PASSWORD}"
    echo "    Gitea (admin)    ${GITEA_ADMIN_USER}            ${GITEA_ADMIN_PASSWORD}"
    echo "    Gitea (DS)       ds-team              ds-team-password"
    echo "    Gitea (DE)       de-team              de-team-password"
    echo "    Woodpecker       (OAuth via Gitea - use any Gitea account)"
    echo "    JupyterHub       admin                ${JUPYTERHUB_PASSWORD}"
    echo "    Grafana          admin                ${GRAFANA_ADMIN_PASSWORD}"
    echo "    Metabase         (first-time setup wizard on first access)"
    echo "    MLflow           (no auth required)"
    echo "    H2O Server       (no auth required)"
    echo "    Zot Registry     (no auth required)"
    echo "    Prometheus       (no auth required)"
    echo ""
    echo "    SeaweedFS S3     Access Key: ${S3_ACCESS_KEY}"
    echo "                     Secret Key: ${S3_SECRET_KEY}"
    echo ""
    echo -e "  ${GREEN}PostgreSQL Databases (host: localhost:30432):${NC}"
    echo ""
    echo "    Database         Username             Password"
    echo "    --------         --------             --------"
    echo "    postgres         postgres             ${PG_ADMIN_PASSWORD}"
    echo "    warehouse        mbt_user             ${PG_MBT_PASSWORD}"
    echo "    mlflow_db        mlflow_user          ${PG_MLFLOW_PASSWORD}"
    echo "    airflow_db       airflow_user         ${PG_AIRFLOW_PASSWORD}"
    echo "    gitea_db         gitea_user           ${PG_GITEA_PASSWORD}"
    echo "    metabase_db      metabase_user        ${PG_METABASE_PASSWORD}"
    echo ""
    echo -e "  ${GREEN}Gitea Repository:${NC}  de-team/ml-pipeline"
    echo ""
    echo -e "  ${GREEN}Kubernetes:${NC}"
    echo "    Provider:     ${K8S_PROVIDER}"
    echo "    Context:      $(kube_context)"
    echo "    Namespaces:   ${NS_MBT}, ${NS_PIPELINES}, ${NS_MONITORING}"
    echo "    Pipeline SA:  mbt-runner (in ${NS_PIPELINES})"
    echo "    MBT Runner:   ${ZOT_HOST}/mbt/mbt-runner:latest"
    echo ""

    if $all_ok; then
        log_success "All services are healthy!"
    else
        log_warn "Some services may still be starting up. Check with: kubectl get pods -n mbt"
    fi
}

# ============================================================
# Teardown
# ============================================================

teardown() {
    log_phase "Tearing down infrastructure ($K8S_PROVIDER)"

    case "$K8S_PROVIDER" in
        k3d)
            log_info "Deleting k3d cluster '$CLUSTER_NAME'..."
            k3d cluster delete "$CLUSTER_NAME" 2>/dev/null || log_warn "Cluster may not exist"
            log_success "Cluster '$CLUSTER_NAME' deleted"
            ;;
        rancher-desktop)
            log_info "Cleaning up MBT resources from Rancher Desktop cluster..."
            kubectl config use-context rancher-desktop 2>/dev/null || true

            # Uninstall Helm releases
            for release in prometheus; do
                helm uninstall "$release" -n "$NS_MONITORING" 2>/dev/null || true
            done
            for release in postgres seaweedfs gitea woodpecker airflow jupyterhub zot-registry; do
                helm uninstall "$release" -n "$NS_MBT" 2>/dev/null || true
            done

            # Delete custom deployments
            for deploy in mlflow h2o metabase; do
                kubectl delete deployment "$deploy" -n "$NS_MBT" --ignore-not-found=true 2>/dev/null || true
            done

            # Delete namespaces (this removes all remaining resources)
            for ns in "$NS_MBT" "$NS_PIPELINES" "$NS_MONITORING"; do
                kubectl delete namespace "$ns" --ignore-not-found=true 2>/dev/null || true
            done

            # Clean up cluster-scoped resources
            kubectl delete clusterrole airflow-pod-operator --ignore-not-found=true 2>/dev/null || true
            kubectl delete clusterrolebinding airflow-pod-operator-binding --ignore-not-found=true 2>/dev/null || true

            log_success "MBT resources removed from Rancher Desktop cluster"
            ;;
    esac

    log_info "Note: Docker images built for Zot are still available locally."
    log_info "Run 'docker image prune' to clean up if needed."
}

# ============================================================
# Status
# ============================================================

status() {
    case "$K8S_PROVIDER" in
        k3d)
            if ! k3d cluster list -o json 2>/dev/null | jq -e ".[] | select(.name==\"$CLUSTER_NAME\")" &>/dev/null; then
                log_error "Cluster '$CLUSTER_NAME' does not exist"
                exit 1
            fi
            k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-switch-context 2>/dev/null
            ;;
        rancher-desktop)
            kubectl config use-context rancher-desktop 2>/dev/null \
                || { log_error "Rancher Desktop context not found"; exit 1; }
            if ! kubectl cluster-info &>/dev/null; then
                log_error "Cannot reach Rancher Desktop cluster"
                exit 1
            fi
            ;;
    esac
    phase_17_summary
}

# ============================================================
# Main
# ============================================================

main() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  MBT Integration Test - Infrastructure Setup (${K8S_PROVIDER})${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""

    local start_time=$SECONDS

    phase_0_prerequisites
    phase_1_cluster
    phase_2_shared_resources
    phase_3_postgres
    phase_4_seaweedfs
    phase_5_zot_registry
    phase_6_build_images
    phase_7_mlflow
    phase_8_gitea
    phase_9_woodpecker
    phase_10_airflow
    phase_11_h2o
    phase_12_jupyterhub
    phase_13_metabase
    phase_14_monitoring
    phase_15_mbt_pipelines_rbac
    phase_16_post_init
    phase_17_summary

    local elapsed=$(( SECONDS - start_time ))
    local minutes=$(( elapsed / 60 ))
    local seconds=$(( elapsed % 60 ))
    echo ""
    log_success "Infrastructure setup completed in ${minutes}m ${seconds}s"
}

# ============================================================
# Entry Point
# ============================================================

case "${1:-}" in
    teardown)
        teardown
        ;;
    status)
        status
        ;;
    *)
        main
        ;;
esac
