#!/usr/bin/env bash
set -euo pipefail
# set -x

install_dir="$(readlink -f "$(dirname "$0")/..")"

export LD_LIBRARY_PATH="$install_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$install_dir/plugins/platforms"
export QT_PLUGIN_PATH="$install_dir/plugins"
export QML2_IMPORT_PATH="$install_dir/qml"

# force XWayland for now...
export QT_QPA_PLATFORM=xcb

set +x

exec "$install_dir/bin/pointcaster-real" "$@"
