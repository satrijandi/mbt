# The root of every showcase image: Canonical's Ubuntu Base rootfs, unpacked
# onto scratch. The docker official `ubuntu` image is built from this same
# tarball; here it comes from cdimage.ubuntu.com instead of a registry, where
# every point release keeps its URL for good:
#   https://cdimage.ubuntu.com/ubuntu-base/releases/<VERSION>/release/
#
# ADD unpacks only a LOCAL archive, so scripts/build_images.sh does the fetch:
# it downloads ubuntu-base-<VERSION>-base-<arch>.tar.gz for the daemon's
# architecture, refuses it unless it matches the ROOTFS_SHA256_* below (from
# that directory's SHA256SUMS), and stages it as rootfs.tar.gz in the build
# context. Nothing here runs, so the image builds in seconds and needs no
# network once the tarball is cached.
#
# It is also the probe image of scripts/docker_sock_gid.sh and doctor.py,
# which need nothing but stat and df.
ARG VERSION=26.04.1
ARG ROOTFS_SHA256_AMD64=a496a960472ce474a59590b8987d3a1135d3cbef1991f3b1abe8cacfea8bf85a
ARG ROOTFS_SHA256_ARM64=5a1906794ced63a71a8119c3f211ef5f0bbe0a243001b4bbd41fdf80c5b219fd

FROM scratch
ADD rootfs.tar.gz /
CMD ["/bin/bash"]
