# Prometheus: the upstream release on the showcase base, with upstream's
# image contract - binaries in /bin, the release's default config, TSDB in
# the /prometheus volume, user nobody. Compose mounts the real config.
ARG VERSION=3.13.3
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:b349c732d8a853e657d0e7ae1bbad4d11b586615fb65fdc59d896b9f869c001e \
    https://github.com/prometheus/prometheus/releases/download/v${VERSION}/prometheus-${VERSION}.linux-amd64.tar.gz \
    /prometheus.tar.gz

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:aa9be9c36608e7ab7cc089726195d0e4fa74f87b330538df9d30d3da1f2b0e18 \
    https://github.com/prometheus/prometheus/releases/download/v${VERSION}/prometheus-${VERSION}.linux-arm64.tar.gz \
    /prometheus.tar.gz

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN --mount=from=fetch,target=/fetch \
    set -eux; \
    mkdir /tmp/prometheus /etc/prometheus /prometheus; \
    tar -xzf /fetch/prometheus.tar.gz -C /tmp/prometheus --strip-components=1; \
    mv /tmp/prometheus/prometheus /tmp/prometheus/promtool /bin/; \
    mv /tmp/prometheus/prometheus.yml /etc/prometheus/; \
    rm -rf /tmp/prometheus; \
    chown nobody:nogroup /prometheus
USER nobody
WORKDIR /prometheus
VOLUME /prometheus
EXPOSE 9090
ENTRYPOINT ["/bin/prometheus"]
CMD ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus"]
