# The node of the k3d + ArgoCD tier (tests/test_showcase_k3d.py), in place
# of rancher/k3s from Docker Hub: k3d runs this image as the cluster's server
# (`k3d cluster create --image`). The release binary is the whole of k3s; it
# carries containerd, runc, CNI and its own iptables, and unpacks them into
# /var/lib/rancher/k3s/data on first use - here at build time, so k3d's node
# entrypoint scripts find iptables-save on PATH before k3s starts, as they do
# in upstream's image. The release's airgap bundle holds every system image
# k3s deploys (pause, coredns, local-path, metrics-server, traefik, klipper),
# which k3s imports from agent/images/ at boot, so the node pulls none.
ARG VERSION=1.35.5
ARG K3S_RELEASE=v1.35.5%2Bk3s1
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG K3S_RELEASE
ADD --checksum=sha256:267d18da7b3c837d82283f0588fb9031a8a6ff3c0dac772c260c40852ce515f6 \
    https://github.com/k3s-io/k3s/releases/download/${K3S_RELEASE}/k3s /k3s
ADD --checksum=sha256:c1a261f664a58664aab975ba5c861d872c825d4f8ccc391c649283bf2c1663cf \
    https://github.com/k3s-io/k3s/releases/download/${K3S_RELEASE}/k3s-airgap-images-amd64.tar.zst \
    /k3s-airgap-images.tar.zst

FROM scratch AS fetch-arm64
ARG K3S_RELEASE
ADD --checksum=sha256:aced90bea6a4e025abef4cf6db7d8cc316a7d20a2b963f809852a37f2b9dbc03 \
    https://github.com/k3s-io/k3s/releases/download/${K3S_RELEASE}/k3s-arm64 /k3s
ADD --checksum=sha256:0bf33db3b0595166f1c36a6780ebd789343f50f23a4d61157519ada4d002919d \
    https://github.com/k3s-io/k3s/releases/download/${K3S_RELEASE}/k3s-airgap-images-arm64.tar.zst \
    /k3s-airgap-images.tar.zst

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
COPY --from=fetch --chmod=0755 /k3s /bin/k3s
COPY --from=fetch --chmod=0644 /k3s-airgap-images.tar.zst /var/lib/rancher/k3s/agent/images/
# k3d's node entrypoint (#!/bin/sh) traps `SIGTERM SIGINT ...`, which the
# busybox sh of upstream's image accepts and dash, Ubuntu's sh, rejects as a
# bad trap - the node then exits on every start. bash accepts both spellings.
RUN set -eux; \
    k3s ctr --version; \
    test -x /var/lib/rancher/k3s/data/current/bin/aux/iptables-save; \
    for tool in kubectl crictl ctr; do ln -s k3s "/bin/$tool"; done; \
    ln -sf bash /usr/bin/sh
ENV PATH=$PATH:/var/lib/rancher/k3s/data/current/bin:/var/lib/rancher/k3s/data/current/bin/aux \
    CRI_CONFIG_FILE=/var/lib/rancher/k3s/agent/etc/crictl.yaml
VOLUME /var/lib/kubelet /var/lib/rancher/k3s /var/lib/cni /var/log
ENTRYPOINT ["/bin/k3s"]
CMD ["agent"]
