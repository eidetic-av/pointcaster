#!/bin/bash
set -euxo pipefail

cmake --preset linux-release

cmake --build build/linux-release

cmake --install build/linux-release --prefix build/linux-release/AppDir/usr

packaging/install-with-linuxdeploy.sh
