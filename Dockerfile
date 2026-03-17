FROM docker.io/zhongruoyu/gcc-ports:15.2-bookworm

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV ARCH=x86_64

ARG CMAKE_VERSION=4.2.1
ARG QT_VERSION=6.11.0
ARG TBB_VERSION=2022.3
ARG CLANG_VERSION=21

ARG APPIMAGETOOL_VERSION=1.9.1
ARG APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"

ARG VCPKG_COMMIT=8eed1d644672846105716dc1c21926b20d928584

ARG DEV_USERNAME=dev
ARG DEV_USER_UID=1000
ARG DEV_USER_GID=1000

# install build toolchain
RUN --mount=type=cache,id=var-cache-apt,target=/var/cache/apt \
    --mount=type=cache,id=var-lib-apt,target=/var/lib/apt \
    set -eux; \
    apt-get update; \
    apt-get -y install --no-install-recommends \
        build-essential ninja-build pkg-config gnupg \
        git curl wget axel ca-certificates tar zip unzip \
        autoconf-archive libtool \
        m4 gettext libltdl-dev \
        bison flex patchelf \
        python3 python3-jinja2 python3-pip; \
    rm -rf /var/lib/apt/lists/*

# platform libs
RUN --mount=type=cache,id=var-cache-apt,target=/var/cache/apt \
    --mount=type=cache,id=var-lib-apt,target=/var/lib/apt \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-dev libegl1-mesa-dev libgles-dev \
        libx11-dev libx11-xcb-dev '^libxcb.*-dev' \
        libglu1-mesa-dev libxrender-dev libxi-dev \
        libxkbcommon-dev libxkbcommon-x11-dev libxext-dev \
        libglib2.0-bin libopengl-dev libglx-dev \
        libfontconfig-dev libfreetype6-dev libdbus-1-dev \
        libxinerama-dev libxcursor-dev libxrandr-dev \
        xorg-dev libglu1-mesa-dev libsystemd-dev \
        libdbus-1-dev libtinfo6 libzstd1 zlib1g; \
    rm -rf /var/lib/apt/lists/*

# download and install cmake
RUN set -eux; \
    BASE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}"; \
    curl -fsSLO "${BASE_URL}/cmake-${CMAKE_VERSION}-SHA-256.txt"; \
    curl -fsSLO "${BASE_URL}/cmake-${CMAKE_VERSION}-linux-x86_64.sh"; \
    grep "cmake-${CMAKE_VERSION}-linux-x86_64.sh" "cmake-${CMAKE_VERSION}-SHA-256.txt" > cmake.sha256; \
    sha256sum -c cmake.sha256; \
    sh "cmake-${CMAKE_VERSION}-linux-x86_64.sh" --skip-license --prefix=/usr/local; \
    rm -f cmake-${CMAKE_VERSION}-linux-x86_64.sh cmake-${CMAKE_VERSION}-SHA-256.txt cmake.sha256

# install clang tools
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates wget gnupg; \
    rm -rf /var/lib/apt/lists/*; \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor > /usr/share/keyrings/llvm-snapshot.gpg; \
    printf "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-${CLANG_VERSION} main\n" > /etc/apt/sources.list.d/llvm${CLANG_VERSION}.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \ 
        clangd-${CLANG_VERSION} clang-tools-${CLANG_VERSION} clang-format-${CLANG_VERSION} clang-tidy-${CLANG_VERSION}; \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-${CLANG_VERSION} 100; \
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-${CLANG_VERSION} 100; \
    update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-${CLANG_VERSION} 100; \
    rm -rf /var/lib/apt/lists/*

# add qt installer (aqt)
RUN --mount=type=cache,id=root-cache-pip,target=/root/.cache/pip \
    set -eux; \
    python3 -m pip install --no-input aqtinstall --break-system-packages

# install onetbb parallel lib and its deps
RUN --mount=type=cache,id=var-cache-apt,target=/var/cache/apt \
    --mount=type=cache,id=var-lib-apt,target=/var/lib/apt \
    set -eux; \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null; \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | tee /etc/apt/sources.list.d/oneAPI.list; \
    apt-get update; \
    apt-get -y install --no-install-recommends \
        hwloc intel-oneapi-tbb-devel-${TBB_VERSION}; \
    rm -rf /var/lib/apt/lists/*

# set up any env vars

ENV USERBIN_DIR=/opt/bin
ENV VCPKG_ROOT=/opt/vcpkg
ENV QT_INSTALL_DIR=/opt/qt
ENV Qt6_DIR=${QT_INSTALL_DIR}/${QT_VERSION}
ENV APPIMAGETOOL_DIR=${USERBIN_DIR}
ENV TBBROOT=/opt/intel/oneapi/tbb/${TBB_VERSION}

ENV PATH="${USERBIN_DIR}:${VCPKG_ROOT}:${Qt6_DIR}/gcc_64/bin:${PATH}"

# set up extra directories for dependency locations that need to be manipulated by the dev user
RUN set -eux; \
    mkdir -p /opt ${USERBIN_DIR}; \
    chown "${DEV_USER_UID}:${DEV_USER_GID}" /opt ${USERBIN_DIR}

# dev niceties
RUN --mount=type=cache,id=var-cache-apt,target=/var/cache/apt \
    --mount=type=cache,id=var-lib-apt,target=/var/lib/apt \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends neovim fish; \
    rm -rf /var/lib/apt/lists/*

# set up dev user
RUN set -eux; \
    groupadd --gid "${DEV_USER_GID}" "${DEV_USERNAME}"; \
    useradd --uid "${DEV_USER_UID}" --gid "${DEV_USER_GID}" -m "${DEV_USERNAME}"

# move to the dev user's environment now and set up the tools that don't require root
USER ${DEV_USERNAME}

WORKDIR /pointcaster

# install qt6 libs
RUN set -eux; \
    aqt install-qt \
        --outputdir "${QT_INSTALL_DIR}" \
        linux desktop ${QT_VERSION} linux_gcc_64 \
        -m qtshadertools qtquick3d qttasktree

ENV CMAKE_PREFIX_PATH="${QT_INSTALL_DIR}/${QT_VERSION}/gcc_64"

# download and bootstrap vcpkg into the user's home dir
ENV VCPKG_DEFAULT_BINARY_CACHE=/home/${DEV_USERNAME}/.cache/vcpkg/archives
ENV VCPKG_DOWNLOADS=${VCPKG_ROOT}-cache/downloads
ENV VCPKG_BUILDTREES=${VCPKG_ROOT}-cache/buildtrees

RUN --mount=type=cache,id=vcpkg-downloads,target=${VCPKG_DOWNLOADS},uid=${DEV_USER_UID},gid=${DEV_USER_GID} \
    --mount=type=cache,id=vcpkg-buildtrees,target=${VCPKG_BUILDTREES},uid=${DEV_USER_UID},gid=${DEV_USER_GID} \
    set -eux; \
    mkdir -p ${VCPKG_DEFAULT_BINARY_CACHE}; \
    git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"; \
    cd "${VCPKG_ROOT}"; \
    git reset --hard "${VCPKG_COMMIT}"; \
    ./bootstrap-vcpkg.sh -disableMetrics

# install appimagetool for AppImage deployment
RUN --mount=type=cache,id=user-downloads,target=/home/${DEV_USERNAME}/.cache/downloads,uid=${DEV_USER_UID},gid=${DEV_USER_GID} \
    set -eux; \
    mkdir -p "${APPIMAGETOOL_DIR}"; \
    cd "${APPIMAGETOOL_DIR}"; \
    test -f "/home/${DEV_USERNAME}/.cache/downloads/appimagetool-x86_64.AppImage" || \
        axel "https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-x86_64.AppImage" \
            -o "/home/${DEV_USERNAME}/.cache/downloads/appimagetool-x86_64.AppImage"; \
    echo "${APPIMAGETOOL_SHA256}  /home/${DEV_USERNAME}/.cache/downloads/appimagetool-x86_64.AppImage" | sha256sum -c -; \
    cp -f "/home/${DEV_USERNAME}/.cache/downloads/appimagetool-x86_64.AppImage" ./appimagetool-x86_64.AppImage; \
    chmod +x appimagetool-x86_64.AppImage; \
    mkdir -p appimagetool && cd appimagetool; \
    ../appimagetool-x86_64.AppImage --appimage-extract
