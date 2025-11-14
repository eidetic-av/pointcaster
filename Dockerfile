# debian trixie from 2025-11-06
FROM debian@sha256:01a723bf5bfb21b9dda0c9a33e0538106e4d02cce8f557e118dd61259553d598

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 

# we use debian snapshot packages so build dependencies provided through
# apt are pinned to a single date also
ARG SNAPSHOT=20251106T204438Z

# freeze apt to the snapshot date by specifying package repos
RUN set -eux; \
	cat > /etc/apt/sources.list.d/debian.sources <<EOF
Types: deb
URIs: http://snapshot.debian.org/archive/debian/${SNAPSHOT}
Suites: trixie trixie-updates
Components: main contrib non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: http://snapshot.debian.org/archive/debian-security/${SNAPSHOT}
Suites: trixie-security
Components: main contrib non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
EOF

RUN set -eux; \
	echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99snapshot-no-expire; \
	apt-get -y update && apt-get -y dist-upgrade

# install build toolchain and system dependencies
RUN set -eux; apt-get -y install --no-install-recommends \
	build-essential g++ cmake ninja-build pkg-config \
	git curl wget ca-certificates tar zip unzip \
  autoconf automake autoconf-archive libtool m4 gettext \
  libltdl-dev bison flex \
  python3 python3-jinja2

RUN set -eux; apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    libx11-dev libx11-xcb-dev libwayland-dev libwayland-egl1 \
    '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev \
    libxrender-dev libxi-dev libxkbcommon-dev \
    libxkbcommon-x11-dev libegl1-mesa-dev libxkbcommon-dev \
    libxi-dev libxext-dev \
    libxrender-dev

# install qt6
RUN set -eux; apt-get -y install --no-install-recommends \
	qt6-base-dev qt6-declarative-dev qt6-shadertools-dev \
	qml6-module-qtquick qml6-module-qtquick-controls \
	qml6-module-qtquick-templates qml6-module-qtquick-layouts \
	qml6-module-qtquick-window qt6-base-private-dev \
	qt6-declarative-private-dev qt6-wayland qt6-wayland-dev \
	libxkbcommon-dev 

# download and bootstrap vcpkg

# the commit hash is the builtin-baseline from manifest.json

ARG VCPKG_COMMIT=1dce60b00170a48594cdb9adbcecf03a26066fec
ENV VCPKG_ROOT=/opt/vcpkg
RUN set -eux; \
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"; \
    cd "$VCPKG_ROOT"; \
    git reset --hard "$VCPKG_COMMIT"; \
    ./bootstrap-vcpkg.sh -disableMetrics; \
    git config --global --add safe.directory "$VCPKG_ROOT"

