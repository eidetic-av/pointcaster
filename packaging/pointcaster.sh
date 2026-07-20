#!/usr/bin/env bash
set -euo pipefail
# set -x

install_dir="$(readlink -f "$(dirname "$0")/..")"

export LD_LIBRARY_PATH="$install_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# prefer the bundled gcc runtime over the host copy only when it's actually newer
# https://github.com/linuxdeploy/linuxdeploy-plugin-checkrt
glibcxx_version() { grep -oa 'GLIBCXX_[0-9.]*' "$1" 2>/dev/null | sort -V | tail -n1 || true; }
host_libstdcxx="$({ PATH="/sbin:/usr/sbin:$PATH" ldconfig -p 2>/dev/null || true; } | awk '/libstdc\+\+\.so\.6 \(libc6,x86-64\)/{print $NF; exit}')"
host_glibcxx="$(glibcxx_version "$host_libstdcxx")"
bundled_glibcxx="$(glibcxx_version "$install_dir/optional/gcc/libstdc++.so.6")"
if [ -n "$bundled_glibcxx" ]; then
  if [ -z "$host_glibcxx" ] || [ "$(printf '%s\n%s\n' "$host_glibcxx" "$bundled_glibcxx" | sort -V | tail -n1)" != "$host_glibcxx" ]; then
    export LD_LIBRARY_PATH="$install_dir/optional/gcc:$LD_LIBRARY_PATH"
    echo "pointcaster: using bundled gcc runtime ($bundled_glibcxx, host has ${host_glibcxx:-none})"
  else
    echo "pointcaster: using host gcc runtime ($host_glibcxx at ${host_libstdcxx:-unknown})"
  fi
fi
export QT_QPA_PLATFORM_PLUGIN_PATH="$install_dir/plugins/platforms"
export QT_PLUGIN_PATH="$install_dir/plugins"
export QML2_IMPORT_PATH="$install_dir/qml"

export PATH="$install_dir/bin:$PATH"

# force X11/XWayland
# - wayland isn't able to restore the state of floating window layouts
# - it always draws application title bars with kddockwidgets
# https://github.com/KDAB/KDDockWidgets/blob/main/docs/book/src/qpa-wayland.md
export QT_QPA_PLATFORM=xcb

set +x

exec "$install_dir/bin/pointcaster" "$@"
