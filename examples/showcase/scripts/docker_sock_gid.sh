#!/bin/sh
# Print the GID of the group owning /var/run/docker.sock AS CONTAINERS SEE
# IT. airflow-scheduler's non-root user joins that group (compose
# group_add) so its DAG tasks can drive the daemon.
#
# The host's own view is the wrong question everywhere but native Linux:
# compose resolves the bind mount inside the daemon's VM, and each runtime
# presents the socket differently there - group 0 on Docker Desktop and
# OrbStack, a `docker` group on Rancher Desktop's Lima VM, the user's own
# group under rootless Docker. Asking the daemon is right on all of them.
#
# The probe image is one the stack already pulls (airflow-db's), so this
# costs no extra download; keep the tag in step with compose. Prints 0 and
# stays quiet when the daemon is unreachable, so `make down` and friends
# still parse; compose then fails on its own, legibly.
set -u

PROBE_IMAGE="postgres:18.6-alpine"

out="$(docker run --rm --network none --entrypoint stat \
    -v /var/run/docker.sock:/var/run/docker.sock \
    "$PROBE_IMAGE" -c '%g %a' /var/run/docker.sock 2>/dev/null)" || out=""

case "$out" in
    *[0-9]" "[0-9]*) ;;
    *) echo 0; exit 0 ;;
esac

gid="${out% *}"
mode="${out#* }"
# Joining the group only helps when the group may read and write: a 0600
# socket (seen on some rootless setups) refuses the scheduler regardless.
case "$mode" in
    *[67][0-7]) ;;
    *) echo "docker_sock_gid: /var/run/docker.sock is mode $mode inside containers;" \
            "airflow-scheduler cannot use it via its group (needs 660)." >&2 ;;
esac
echo "$gid"
