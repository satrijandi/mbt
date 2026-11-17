# Grafana: the upstream release tarball on the showcase base, laid out like
# upstream's image - home /usr/share/grafana, the release's sample.ini as
# /etc/grafana/grafana.ini, data under /var/lib/grafana, user grafana (472,
# group 0) - and started with the arguments upstream's run.sh passes.
# Compose mounts the provisioning and the dashboards.
ARG VERSION=13.1.5
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:35d3f3b93777564a95b6f2f9380340ac07b0f238e2df6ba46a106fde785d61f7 \
    https://dl.grafana.com/oss/release/grafana-${VERSION}.linux-amd64.tar.gz /grafana.tar.gz

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:6d6f9a434803fcbc55a3adb344e4f6c821db70791ab47449d6d7cfc5fd60e41e \
    https://dl.grafana.com/oss/release/grafana-${VERSION}.linux-arm64.tar.gz /grafana.tar.gz

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN apt-get install -y --no-install-recommends tzdata && apt-get clean
RUN --mount=from=fetch,target=/fetch \
    set -eux; \
    mkdir -p /usr/share/grafana /etc/grafana/provisioning/datasources \
        /etc/grafana/provisioning/dashboards /var/lib/grafana/plugins /var/log/grafana; \
    tar -xzf /fetch/grafana.tar.gz -C /usr/share/grafana --strip-components=1; \
    cp /usr/share/grafana/conf/sample.ini /etc/grafana/grafana.ini; \
    cp /usr/share/grafana/conf/ldap.toml /etc/grafana/ldap.toml; \
    useradd --uid 472 --gid 0 --home-dir /usr/share/grafana --no-create-home --shell /usr/sbin/nologin grafana; \
    chown -R grafana:0 /var/lib/grafana /var/log/grafana /etc/grafana/provisioning; \
    chmod -R 777 /var/lib/grafana /var/log/grafana /etc/grafana/provisioning

ENV PATH=/usr/share/grafana/bin:$PATH \
    GF_PATHS_CONFIG=/etc/grafana/grafana.ini \
    GF_PATHS_DATA=/var/lib/grafana \
    GF_PATHS_HOME=/usr/share/grafana \
    GF_PATHS_LOGS=/var/log/grafana \
    GF_PATHS_PLUGINS=/var/lib/grafana/plugins \
    GF_PATHS_PROVISIONING=/etc/grafana/provisioning \
    HOME=/usr/share/grafana
USER 472
WORKDIR /usr/share/grafana
EXPOSE 3000
ENTRYPOINT ["grafana", "server", "--homepath=/usr/share/grafana", "--config=/etc/grafana/grafana.ini", \
    "--packaging=docker", "cfg:default.log.mode=console", "cfg:default.paths.data=/var/lib/grafana", \
    "cfg:default.paths.logs=/var/log/grafana", "cfg:default.paths.plugins=/var/lib/grafana/plugins", \
    "cfg:default.paths.provisioning=/etc/grafana/provisioning"]
