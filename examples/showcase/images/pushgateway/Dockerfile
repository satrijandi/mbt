# Pushgateway: the upstream release on the showcase base, with upstream's
# image contract - /bin/pushgateway, user nobody, workdir /pushgateway.
ARG VERSION=1.11.3
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:bb0a44dee0953df9e8cd3c082981ff50327de56d965d83bdd9b0957d83921e38 \
    https://github.com/prometheus/pushgateway/releases/download/v${VERSION}/pushgateway-${VERSION}.linux-amd64.tar.gz \
    /pushgateway.tar.gz

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:727ff0098943657b44c21a029be9d9fcc4f249ec72dcb9f0a34aa66b2d5f1ecc \
    https://github.com/prometheus/pushgateway/releases/download/v${VERSION}/pushgateway-${VERSION}.linux-arm64.tar.gz \
    /pushgateway.tar.gz

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN --mount=from=fetch,target=/fetch \
    tar -xzf /fetch/pushgateway.tar.gz -C /bin --strip-components=1 --wildcards '*/pushgateway' \
    && mkdir /pushgateway \
    && chown nobody:nogroup /pushgateway
USER nobody
WORKDIR /pushgateway
EXPOSE 9091
ENTRYPOINT ["/bin/pushgateway"]
