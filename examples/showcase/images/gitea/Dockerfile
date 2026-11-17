# Gitea (the forge): the upstream release binary on the showcase base, laid
# out like upstream's rootless image - user git (1000:1000), state under
# /var/lib/gitea, Gitea's own ssh server on 2222 - with entrypoint.sh doing
# what that image's docker-setup.sh did for compose's GITEA__* variables.
# git is the one runtime dependency; curl serves compose's healthcheck.
ARG VERSION=1.27.3
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:4da93c2c10b6980c359bcb86d5573ebfd7770e2e151756534edee24c8c12d971 \
    https://github.com/go-gitea/gitea/releases/download/v${VERSION}/gitea-${VERSION}-linux-amd64 /gitea

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:04c086d36dba793546e331484a9da34571763efdfa77dc526cc98e0f10917e7b \
    https://github.com/go-gitea/gitea/releases/download/v${VERSION}/gitea-${VERSION}-linux-arm64 /gitea

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN apt-get install -y --no-install-recommends git curl dumb-init \
    && apt-get clean \
    && groupadd --gid 1000 git \
    && useradd --uid 1000 --gid git --home-dir /var/lib/gitea/git --no-create-home --shell /bin/bash git \
    && mkdir -p /var/lib/gitea \
    && chown git:git /var/lib/gitea
COPY --from=fetch --chmod=0755 /gitea /usr/local/bin/gitea
COPY app.ini /usr/local/share/gitea/app.ini
COPY --chmod=0755 entrypoint.sh /usr/local/bin/gitea-entrypoint

ENV GITEA_WORK_DIR=/var/lib/gitea \
    GITEA_CUSTOM=/var/lib/gitea/custom \
    GITEA_TEMP=/tmp/gitea \
    TMPDIR=/tmp/gitea \
    HOME=/var/lib/gitea/git
USER 1000:1000
WORKDIR /var/lib/gitea
VOLUME /var/lib/gitea
EXPOSE 2222 3000
ENTRYPOINT ["dumb-init", "--", "/usr/local/bin/gitea-entrypoint"]
CMD ["gitea", "web"]
