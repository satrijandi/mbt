# Woodpecker's clone plugin, which every showcase pipeline names in its
# `clone:` block: the upstream release binary on the showcase base, with
# upstream's image contract - git and git-lfs, PLUGIN_HOME=/app, the binary
# as entrypoint. woodpecker-server trusts this image as its default clone
# plugin (WOODPECKER_DEFAULT_CLONE_PLUGIN in compose).
ARG VERSION=2.10.0
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:be3c3fa5363ddaff0dac88c2dc5b745ca785f21d5554f90b47d132d34baebcec \
    https://github.com/woodpecker-ci/plugin-git/releases/download/${VERSION}/linux-amd64_plugin-git /plugin-git

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:61c9f28d70b4eba433d3abcebfe00b177243fc39ed0da1cd4d7d308bb0588782 \
    https://github.com/woodpecker-ci/plugin-git/releases/download/${VERSION}/linux-arm64_plugin-git /plugin-git

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN apt-get install -y --no-install-recommends git git-lfs openssh-client \
    && apt-get clean \
    && mkdir /app
COPY --from=fetch --chmod=0755 /plugin-git /bin/plugin-git
ENV GODEBUG=netdns=go \
    PLUGIN_HOME=/app
ENTRYPOINT ["/bin/plugin-git"]
