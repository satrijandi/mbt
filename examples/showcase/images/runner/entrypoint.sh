#!/bin/sh
# Preflight for every mbt-running container (DESIGN.md section 2): the Spark
# adapters stage split writes through driver-local temp dirs that executors
# must also see, so a missing/unwritable shared TMPDIR fails distantly with
# "split ... materialized 0 rows". Fail fast and legibly here instead.
set -eu

# Spark standalone runs Python drivers client-mode: executors connect BACK
# to the driver, so every container that may drive the prod target
# advertises its own address (profiles.yml reads SPARK_DRIVER_HOST).
if [ -z "${SPARK_DRIVER_HOST:-}" ]; then
    SPARK_DRIVER_HOST="$(hostname -i 2>/dev/null | cut -d' ' -f1 || true)"
    export SPARK_DRIVER_HOST
fi

if [ -n "${TMPDIR:-}" ]; then
    mkdir -p "$TMPDIR" 2>/dev/null || true
    if [ ! -w "$TMPDIR" ]; then
        echo "showcase preflight: TMPDIR=$TMPDIR is not writable." >&2
        echo "The shared /workspace volume must be mounted into every mbt/Spark container." >&2
        exit 1
    fi
fi

exec "$@"
