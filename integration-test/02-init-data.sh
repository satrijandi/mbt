#!/usr/bin/env bash
# ============================================================================
# 02-init-data.sh - Initialize warehouse data in PostgreSQL
#
# Creates the churn prediction data model and loads realistic seed data
# into the warehouse database. Data is accessible to both mbt_user (pipelines)
# and metabase_user (BI dashboards).
#
# Tables created:
#   - customers_to_score  (customer_id, snapshot_date)
#   - features_a          (customer_id, snapshot_date, feature_a..feature_z)
#   - features_b          (customer_id, snapshot_date, feature_1..feature_1000)
#   - labels              (customer_id, snapshot_date, is_churn)
#   - churn_predictions   (output table for serving pipeline)
#
# Prerequisites:
#   - PostgreSQL running and accessible (k3d cluster or local)
#   - Python 3.10+ available
#   - psql client installed
#
# Usage:
#   ./02-init-data.sh                          # Uses defaults (localhost:30432)
#   PGHOST=myhost PGPORT=5432 ./02-init-data.sh  # Custom connection
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Configuration ────────────────────────────────────────────────────────────
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-30432}"
PGUSER="${PGUSER:-postgres}"
PGPASSWORD="${PGPASSWORD:-admin_password}"
export PGPASSWORD

DB_WAREHOUSE="warehouse"
DATA_DIR=$(mktemp -d)
trap 'rm -rf "$DATA_DIR"' EXIT

# ── Helpers ──────────────────────────────────────────────────────────────────
log_info()    { echo "[$(date +%H:%M:%S)] INFO  $*"; }
log_success() { echo "[$(date +%H:%M:%S)] OK    $*"; }
log_error()   { echo "[$(date +%H:%M:%S)] ERROR $*" >&2; }

psql_warehouse() {
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$DB_WAREHOUSE" \
        -v ON_ERROR_STOP=1 --no-psqlrc -q "$@"
}

# ── Preflight checks ────────────────────────────────────────────────────────
log_info "Checking prerequisites..."

if ! command -v psql &>/dev/null; then
    log_error "psql not found. Install postgresql-client."
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    log_error "python3 not found."
    exit 1
fi

# Test connection
if ! psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$DB_WAREHOUSE" \
        -c "SELECT 1;" &>/dev/null; then
    log_error "Cannot connect to PostgreSQL at $PGHOST:$PGPORT/$DB_WAREHOUSE"
    log_error "Is the database running? Try: kubectl port-forward -n mbt svc/postgres-postgresql 30432:5432"
    exit 1
fi

log_success "Connected to PostgreSQL ($PGHOST:$PGPORT/$DB_WAREHOUSE)"

# ── Step 1: Generate CSV data ───────────────────────────────────────────────
log_info "Generating seed data..."
python3 "$SCRIPT_DIR/infra/postgres/generate_data.py" --output-dir "$DATA_DIR"
log_success "CSV data generated in $DATA_DIR"

# ── Step 2: Create tables (DDL) ─────────────────────────────────────────────
log_info "Creating warehouse tables..."
psql_warehouse -f "$DATA_DIR/ddl.sql"
log_success "Tables created"

# ── Step 3: Load data via COPY ───────────────────────────────────────────────
for table in customers_to_score features_a features_b labels; do
    row_count=$(wc -l < "$DATA_DIR/${table}.csv")
    row_count=$((row_count - 1))  # subtract header
    log_info "Loading $table ($row_count rows)..."
    psql_warehouse -c "\\copy $table FROM '$DATA_DIR/${table}.csv' WITH (FORMAT csv, HEADER true)"
done

log_success "All data loaded"

# ── Step 4: Verify ───────────────────────────────────────────────────────────
log_info "Verifying data..."
echo ""

psql_warehouse -c "
SELECT
    'customers_to_score' AS table_name,
    COUNT(*) AS row_count,
    COUNT(DISTINCT customer_id) AS distinct_customers,
    COUNT(DISTINCT snapshot_date) AS distinct_snapshots
FROM customers_to_score
UNION ALL
SELECT
    'features_a',
    COUNT(*),
    COUNT(DISTINCT customer_id),
    COUNT(DISTINCT snapshot_date)
FROM features_a
UNION ALL
SELECT
    'features_b',
    COUNT(*),
    COUNT(DISTINCT customer_id),
    COUNT(DISTINCT snapshot_date)
FROM features_b
UNION ALL
SELECT
    'labels',
    COUNT(*),
    COUNT(DISTINCT customer_id),
    COUNT(DISTINCT snapshot_date)
FROM labels;
"

echo ""
psql_warehouse -c "
SELECT
    'Churn rate' AS metric,
    ROUND(100.0 * SUM(is_churn) / COUNT(*), 1) || '%' AS value
FROM labels
UNION ALL
SELECT
    'features_b columns',
    COUNT(column_name)::TEXT
FROM information_schema.columns
WHERE table_name = 'features_b' AND column_name LIKE 'feature_%';
"

echo ""
log_info "Testing metabase_user access..."
PGPASSWORD=metabase_password psql -h "$PGHOST" -p "$PGPORT" \
    -U metabase_user -d "$DB_WAREHOUSE" --no-psqlrc -q -c \
    "SELECT COUNT(*) AS metabase_can_read FROM customers_to_score;" 2>/dev/null \
    && log_success "metabase_user can read warehouse tables" \
    || log_error "metabase_user cannot access warehouse tables"

echo ""
log_success "Warehouse data initialization complete!"
log_info "Tables: customers_to_score, features_a, features_b, labels, churn_predictions"
log_info "Access: mbt_user (read/write), metabase_user (read-only)"
