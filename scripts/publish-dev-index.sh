#!/usr/bin/env bash
# Rebuild the multi-platform 'latest' index for the pointcaster-dev image
#
# Runs after both publish-dev-image.sh and publish-dev-image.ps1 have completed

set -euo pipefail

IMAGE=flowbox:5000/pointcaster-dev
PLATFORM_TAGS=(latest-linux latest-windows)

missing=()
for tag in "${PLATFORM_TAGS[@]}"; do
  if ! crane digest --insecure "${IMAGE}:${tag}" >/dev/null 2>&1; then
    missing+=("${IMAGE}:${tag}")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  printf 'error: %s not found\n' "${missing[@]}" >&2
  echo 'the per-platform publish scripts must both complete before indexing' >&2
  exit 1
fi

args=()
for tag in "${PLATFORM_TAGS[@]}"; do
  args+=(-m "${IMAGE}:${tag}")
done

echo "==> index ${IMAGE}:latest (${PLATFORM_TAGS[*]})"
crane index append --insecure "${args[@]}" -t "${IMAGE}:latest"

echo "==> done: ${IMAGE}:latest"
