# Zot (the OCI registry for deployable units and provenance): the upstream
# release binary on the showcase base, serving config.json beside this file.
# That is upstream's image config minus `search.cve`, whose scanner downloads
# its vulnerability database from ghcr.io in the background - the one pull
# from a registry the stack would still have made. Search, the UI and mgmt
# stay on.
#
# The v2.1.16 pin is load-bearing: see compose/docker-compose.yml.
ARG VERSION=2.1.16
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:f9b7e805056db94c97d34bcff4eab77ec8e7cdc2cb76fd6db3d7ad7143605dba \
    https://github.com/project-zot/zot/releases/download/v${VERSION}/zot-linux-amd64 /zot

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:43ccc539feb4be3f86bdca1c16b3d2df3bba7cea475930c4293339851d55b089 \
    https://github.com/project-zot/zot/releases/download/v${VERSION}/zot-linux-arm64 /zot

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
COPY --from=fetch --chmod=0755 /zot /usr/local/bin/zot
COPY config.json /etc/zot/config.json
EXPOSE 5000
ENTRYPOINT ["/usr/local/bin/zot"]
CMD ["serve", "/etc/zot/config.json"]
