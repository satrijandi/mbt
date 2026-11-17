# ArgoCD for the k3d + ArgoCD tier, in place of quay.io/argoproj/argocd:
# the release binary - one multi-call binary, the same ./cmd build the image
# ships, which picks its component from the name it runs as - on the
# showcase base, with the runtime stage of upstream's Dockerfile (itself
# Ubuntu-based) reproduced: user argocd (999), its packages (gpg with its
# agent: the repo-server generates a signing key at start), the gpg and
# git-verify wrappers from hack/, the /app/config mount points, the
# component symlinks, tini as PID 1. helm and kustomize stay out: the tier
# syncs a plain directory of manifests.
ARG VERSION=3.4.9
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:4da1d99cba5fb9158c46c6c2a8298bd9352be93dc7aad935eaefba735c6f4c19 \
    https://github.com/argoproj/argo-cd/releases/download/v${VERSION}/argocd-linux-amd64 /argocd

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:8e486921d301988f2ce725ca37970bd6518cddeaba62d76189476c3ad3f667ea \
    https://github.com/argoproj/argo-cd/releases/download/v${VERSION}/argocd-linux-arm64 /argocd

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
ARG VERSION
RUN set -eux; \
    groupadd --gid 999 argocd; \
    useradd --system --uid 999 --gid argocd argocd; \
    mkdir -p /home/argocd; \
    chown argocd:0 /home/argocd; \
    chmod g=u /home/argocd; \
    apt-get install -y --no-install-recommends \
        git git-lfs tini gpg gpg-agent tzdata connect-proxy openssh-client; \
    apt-get clean
ADD --chmod=0755 --checksum=sha256:06f3503b80ff2c9753c599fe4afdd21e36d1cec7b95244186fc06c2bd96d7180 \
    https://raw.githubusercontent.com/argoproj/argo-cd/v${VERSION}/hack/gpg-wrapper.sh /usr/local/bin/
ADD --chmod=0755 --checksum=sha256:c8f193fcb4f9850dddd0b9c536eae29581d2dd96226cd37d9bb682bb9e5447f8 \
    https://raw.githubusercontent.com/argoproj/argo-cd/v${VERSION}/hack/git-verify-wrapper.sh /usr/local/bin/
RUN set -eux; \
    mkdir -p /app/config/ssh /app/config/tls /app/config/gpg/source /app/config/gpg/keys; \
    touch /app/config/ssh/ssh_known_hosts; \
    ln -sf /app/config/ssh/ssh_known_hosts /etc/ssh/ssh_known_hosts; \
    chown argocd /app/config/gpg/keys; \
    chmod 0700 /app/config/gpg/keys
COPY --from=fetch --chmod=0755 /argocd /usr/local/bin/argocd
RUN for component in server repo-server cmp-server application-controller dex notifications \
        applicationset-controller k8s-auth commit-server; do \
        ln -s argocd "/usr/local/bin/argocd-$component"; \
    done

ENV USER=argocd \
    GRPC_ENABLE_TXT_SERVICE_CONFIG=false
USER 999
WORKDIR /home/argocd
ENTRYPOINT ["/usr/bin/tini", "--"]
