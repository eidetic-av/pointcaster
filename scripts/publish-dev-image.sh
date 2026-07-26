#!/usr/bin/env bash
# Build and publish the Linux pointcaster-dev container image

set -euo pipefail

IMAGE=flowbox:5000/pointcaster-dev
DATE="$(date -u +%Y-%m-%d)"
TAG="${DATE}-linux"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> build ${IMAGE}:${TAG}"
docker build -f Dockerfile -t "${IMAGE}:${TAG}" \
  --label org.opencontainers.image.version="${DATE}" \
  --label org.opencontainers.image.revision="$(git rev-parse HEAD)" \
  --label org.opencontainers.image.created="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$@" .

# docker push seems to fail with the container registry behind cloudflare tunnel...
# so save to a tarball first so crane can upload from a local file
TAR="/var/tmp/pointcaster-dev-${TAG}.tar"
trap 'rm -f "${TAR}"' EXIT

echo "==> save"
docker save "${IMAGE}:${TAG}" -o "${TAR}"

echo "==> push"
crane push --insecure "${TAR}" "${IMAGE}:${TAG}"

echo "==> done: ${IMAGE}:${TAG}"