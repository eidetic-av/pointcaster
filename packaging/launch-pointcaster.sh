#!/usr/bin/env bash
set -euo pipefail
# set -x

install_dir="$(readlink -f "$(dirname "$0")/..")"

export LD_LIBRARY_PATH="$install_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$install_dir/plugins/platforms"
export QT_PLUGIN_PATH="$install_dir/plugins"
export QML2_IMPORT_PATH="$install_dir/qml"

# force X11/XWayland
# - wayland isn't able to restore the state of floating window layouts
# - it always draws application title bars with kddockwidgets
# https://github.com/KDAB/KDDockWidgets/blob/main/docs/book/src/qpa-wayland.md
export QT_QPA_PLATFORM=xcb

set +x

exec "$install_dir/bin/pointcaster-real" "$@"
