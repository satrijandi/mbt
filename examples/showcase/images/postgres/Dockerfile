# PostgreSQL (Airflow's metadata database): Ubuntu's own postgresql-18 build
# from the base image's archive snapshot, run by the subset of the docker
# official image's init that compose uses (entrypoint.sh: POSTGRES_USER,
# POSTGRES_PASSWORD, POSTGRES_DB). Same major, minor, PGDATA and uid (70) as
# the postgres:18.6-alpine image it replaces; the cluster's locale is
# C.UTF-8, which needs no locale data in the image.
ARG VERSION=18.6

FROM localhost/mbt-showcase-base:26.04.1
ARG VERSION
# postgresql-common would create and start a `main` cluster at install time;
# the entrypoint makes the only cluster, on first start, in PGDATA.
RUN set -eux; \
    groupadd --system --gid 70 postgres; \
    useradd --system --uid 70 --gid postgres --home-dir /var/lib/postgresql --shell /bin/bash postgres; \
    apt-get install -y --no-install-recommends postgresql-common; \
    sed -ri 's/#(create_main_cluster) .*$/\1 = false/' /etc/postgresql-common/createcluster.conf; \
    apt-get install -y --no-install-recommends "postgresql-18=${VERSION}-*"; \
    apt-get clean; \
    mkdir -p /var/lib/postgresql /var/run/postgresql; \
    chown -R postgres:postgres /var/lib/postgresql /var/run/postgresql
COPY --chmod=0755 entrypoint.sh /usr/local/bin/postgres-entrypoint

ENV PATH=/usr/lib/postgresql/18/bin:$PATH \
    PGDATA=/var/lib/postgresql/18/docker
USER postgres
VOLUME /var/lib/postgresql
EXPOSE 5432
STOPSIGNAL SIGINT
ENTRYPOINT ["/usr/local/bin/postgres-entrypoint"]
