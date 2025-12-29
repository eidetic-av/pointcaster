#!/bin/bash
set -euxo pipefail
cmake --preset linux-release
cmake --build build/linux-release --target package
