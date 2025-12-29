# debian trixie from 2025-11-06
FROM debian@sha256:01a723bf5bfb21b9dda0c9a33e0538106e4d02cce8f557e118dd61259553d598

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 
ENV ARCH=x86_64

# we use debian snapshot packages so build dependencies provided through
# apt are pinned to a single date

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

# install build toolchain

RUN set -eux; apt-get -y install --no-install-recommends \
	build-essential g++ ninja-build pkg-config \
	git curl wget ca-certificates tar zip unzip \
	autoconf automake autoconf-archive libtool m4 gettext \
	libltdl-dev bison flex patchelf \
	python3 python3-jinja2 python3-pip

# download and install cmake

ARG CMAKE_VERSION=4.2.1
RUN set -eux; \
    BASE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}"; \
    curl -fsSLO "${BASE_URL}/cmake-${CMAKE_VERSION}-SHA-256.txt"; \
    curl -fsSLO "${BASE_URL}/cmake-${CMAKE_VERSION}-linux-x86_64.sh"; \
    grep "cmake-${CMAKE_VERSION}-linux-x86_64.sh" "cmake-${CMAKE_VERSION}-SHA-256.txt" > cmake.sha256; \
    sha256sum -c cmake.sha256; \
    sh "cmake-${CMAKE_VERSION}-linux-x86_64.sh" --skip-license --prefix=/usr/local

# platform libs

RUN set -eux; apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    libx11-dev libx11-xcb-dev libwayland-dev libwayland-egl1 \
    '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev \
    libxrender-dev libxi-dev libxkbcommon-dev \
    libxkbcommon-x11-dev libegl1-mesa-dev libxkbcommon-dev \
    libxi-dev libxext-dev libglib2.0-bin libopengl-dev \
    libxrender-dev libxkbcommon-dev libglx-dev \
    libfontconfig-dev libfreetype6-dev libdbus-1-dev

# add qt installer

RUN set -eux; pip install --no-input aqtinstall --break-system-packages

# set up dev user

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# install some dev niceities

RUN apt-get install -y vim fish

# from here on, run as dev

USER ${USERNAME}
WORKDIR /pointcaster

# install qt6 libs

ARG QT_INSTALL_DIR=/home/${USERNAME}/.local/share/qt
RUN set -eux; aqt install-qt \
        --outputdir "${QT_INSTALL_DIR}" \
        linux desktop 6.11.0 linux_gcc_64 \
        -m qtshadertools
ENV CMAKE_PREFIX_PATH="${QT_INSTALL_DIR}/6.11.0/gcc_64"

# download and bootstrap vcpkg into the user's home dir

ARG VCPKG_COMMIT=1dce60b00170a48594cdb9adbcecf03a26066fec
ENV VCPKG_ROOT=/home/${USERNAME}/.local/share/vcpkg

RUN set -eux; \
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"; \
    cd "$VCPKG_ROOT"; \
    git reset --hard "$VCPKG_COMMIT"; \
    ./bootstrap-vcpkg.sh -disableMetrics; \
    git config --global --add safe.directory "$VCPKG_ROOT"

# install appimagetool for AppImage deployment

ARG APPIMAGE_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
RUN set -eux; \
    APPIMAGE_DIR="/home/${USERNAME}/.local/bin"; \
    mkdir -p "${APPIMAGE_DIR}"; \
    cd ${APPIMAGE_DIR}; \
    curl -fsSLo appimagetool-x86_64.AppImage \
      "https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"; \
    echo "${APPIMAGE_SHA256}  appimagetool-x86_64.AppImage" | sha256sum -c -; \
    chmod +x appimagetool-x86_64.AppImage; \
    mkdir appimagetool && cd appimagetool; \
    ../appimagetool-x86_64.AppImage --appimage-extract
   
