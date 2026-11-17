#!/bin/sh
# The part of the docker official postgres image's entrypoint compose relies
# on: on first start, initdb a cluster whose superuser is POSTGRES_USER with
# POSTGRES_PASSWORD, accept password (scram-sha-256) logins from any host,
# and create POSTGRES_DB; then run postgres listening on every interface.
set -eu

if [ ! -s "$PGDATA/PG_VERSION" ]; then
    mkdir -p "$PGDATA"
    chmod 0700 "$PGDATA"
    pwfile="$(mktemp)"
    printf '%s\n' "$POSTGRES_PASSWORD" > "$pwfile"
    initdb --username="$POSTGRES_USER" --pwfile="$pwfile" --auth-local=trust \
        --auth-host=scram-sha-256 --encoding=UTF8 --locale=C.UTF-8 -D "$PGDATA"
    rm -f "$pwfile"
    echo "host all all all scram-sha-256" >> "$PGDATA/pg_hba.conf"
    if [ "$POSTGRES_DB" != postgres ]; then
        pg_ctl -D "$PGDATA" -o "-c listen_addresses=''" -w start
        psql -v ON_ERROR_STOP=1 --username="$POSTGRES_USER" --dbname=postgres \
            --set=db="$POSTGRES_DB" <<'SQL'
CREATE DATABASE :"db";
SQL
        pg_ctl -D "$PGDATA" -m fast -w stop
    fi
fi

exec postgres -c listen_addresses='*' "$@"
