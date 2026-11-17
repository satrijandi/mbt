# Alertmanager: the upstream release on the showcase base, with upstream's
# image contract - binaries in /bin, the release's default config, state in
# the /alertmanager volume, user nobody. Compose mounts the real config.
ARG VERSION=0.34.1
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:265b9d1e55ef0d5306a436018af6d2b686c2ce051f03d968f7464ecb1372a7e8 \
    https://github.com/prometheus/alertmanager/releases/download/v${VERSION}/alertmanager-${VERSION}.linux-amd64.tar.gz \
    /alertmanager.tar.gz

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:d98d6cbaf52151c7e76e24355fec88b11cebcb9875d4cdd8b76ddce7a7e5535c \
    https://github.com/prometheus/alertmanager/releases/download/v${VERSION}/alertmanager-${VERSION}.linux-arm64.tar.gz \
    /alertmanager.tar.gz

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN --mount=from=fetch,target=/fetch \
    set -eux; \
    mkdir /tmp/alertmanager /etc/alertmanager /alertmanager; \
    tar -xzf /fetch/alertmanager.tar.gz -C /tmp/alertmanager --strip-components=1; \
    mv /tmp/alertmanager/alertmanager /tmp/alertmanager/amtool /bin/; \
    mv /tmp/alertmanager/alertmanager.yml /etc/alertmanager/; \
    rm -rf /tmp/alertmanager; \
    chown nobody:nogroup /alertmanager
USER nobody
WORKDIR /alertmanager
VOLUME /alertmanager
EXPOSE 9093
ENTRYPOINT ["/bin/alertmanager"]
CMD ["--config.file=/etc/alertmanager/alertmanager.yml", "--storage.path=/alertmanager"]
