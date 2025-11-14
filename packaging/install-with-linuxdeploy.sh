#!/usr/bin/env bash
set -euo pipefail

export QML_MODULES_PATHS="AppDir/usr/qml"
export QML_SOURCES_PATHS="AppDir/usr/qml"

export EXTRA_QT_PLUGINS="waylandcompositor"
export EXTRA_PLATFORM_PLUGINS="libqwayland-egl.so;libqwayland-generic.so"

pushd ./build/linux-release

LINUXDEPLOY_SHA256=c20cd71e3a4e3b80c3483cef793cda3f4e990aca14014d23c544ca3ce1270b4d
LINUXDEPLOY_QT_SHA256=15106be885c1c48a021198e7e1e9a48ce9d02a86dd0a1848f00bdbf3c1c92724

wget https://github.com/linuxdeploy/linuxdeploy/releases/download/1-alpha-20251107-1/linuxdeploy-x86_64.AppImage
echo "${LINUXDEPLOY_SHA256}  linuxdeploy-x86_64.AppImage" | sha256sum -c -
chmod +x ./linuxdeploy-x86_64.AppImage

wget https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/1-alpha-20250213-1/linuxdeploy-plugin-qt-x86_64.AppImage
echo "${LINUXDEPLOY_QT_SHA256}  linuxdeploy-plugin-qt-x86_64.AppImage" | sha256sum -c -
chmod +x ./linuxdeploy-plugin-qt-x86_64.AppImage

export APPIMAGE_EXTRACT_AND_RUN=1

./linuxdeploy-x86_64.AppImage \
	--appdir=AppDir \
	--executable=AppDir/usr/bin/pointcaster \
	--desktop-file=../../packaging/pointcaster.desktop \
	--icon-file=../../packaging/pointcaster.png \
	--plugin qt

# patch AppRun to run all arbitrary hook files in
# /apprun-hooks/ to allow easy extension
sed -i '
/\/apprun-hooks\//d
/exec "\$this_dir"\/AppRun\.wrapped "\$@"/ i\
for hook in "$this_dir"/apprun-hooks/*; do\
    [ -f "$hook" ] && source "$hook"\
done\
\
' AppDir/AppRun

popd

# copy environment variable setup for our run env into
# our apprun hook so we can run the executable anywhere
cp ./packaging/run-env.sh ./build/linux-release/AppDir/apprun-hooks/

