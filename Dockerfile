FROM docker.io/zhongruoyu/gcc-ports:15-bookworm

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV ARCH=x86_64

ARG DEV_USER_UID=1000
ARG DEV_USER_GID=1000

# install build toolchain & surrounding tools
RUN --mount=type=cache,id=var-cache-apt,target=/var/cache/apt \
    --mount=type=cache,id=var-lib-apt,target=/var/lib/apt \
    set -eux; \
    apt-get update; \
    apt-get -y install --no-install-recommends \
        build-essential ninja-build pkg-config gnupg \
        git curl wget axel ca-certificates jq \
        tar zip unzip 7zip \
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
ARG CMAKE_VERSION=4.4.0
RUN set -eux; \
    BASE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}"; \
    curl -fsSLO "${BASE_URL}/cmake-${CMAKE_VERSION}-SHA-256.txt"; \
    curl -fsSLO "${BASE_URL}/cmake-${CMAKE_VERSION}-linux-x86_64.sh"; \
    grep "cmake-${CMAKE_VERSION}-linux-x86_64.sh" "cmake-${CMAKE_VERSION}-SHA-256.txt" > cmake.sha256; \
    sha256sum -c cmake.sha256; \
    sh "cmake-${CMAKE_VERSION}-linux-x86_64.sh" --skip-license --prefix=/usr/local; \
    rm -f cmake-${CMAKE_VERSION}-linux-x86_64.sh cmake-${CMAKE_VERSION}-SHA-256.txt cmake.sha256

# install clang tools
ARG CLANG_VERSION=22
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
ARG TBB_VERSION=2023.1
ENV TBB_DIR=/opt/intel/oneapi/tbb/${TBB_VERSION}

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

# install cuda libs
ARG CUDA_VERSION=13.2.1
ARG CUDA_INSTALLER=cuda_13.2.1_595.58.03_linux.run
ARG CUDA_INSTALLER_SHA256="5514a3fe7bcea92b25073c7c100c3e64e7961a7e1dbad6955adb8b59806053f0"

RUN --mount=type=cache,id=root-download-cache,target=/root/.cache/downloads \
    set -eux; \
    test -f /root/.cache/downloads/${CUDA_INSTALLER} || \
        axel "https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_INSTALLER}" \
            -qo /root/.cache/downloads/${CUDA_INSTALLER}; \
    echo "${CUDA_INSTALLER_SHA256} /root/.cache/downloads/${CUDA_INSTALLER}" | sha256sum -c -; \
    sh /root/.cache/downloads/${CUDA_INSTALLER} --toolkit --silent

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDACXX="/usr/local/cuda/bin/nvcc"
ENV Thrust_DIR="/usr/local/cuda/lib64/cmake/thrust"

# before building anything from source, we need to make sure that the libstdc++ version 
# being prioritised is from gcc-ports' gcc 15, not the system bookworm version
RUN set -eux; echo "/usr/local/lib64" > /etc/ld.so.conf.d/gcc15.conf && ldconfig


# set up directories for dependency locations that need to be manipulated by the dev user
ENV USERBIN_DIR=/opt/bin

RUN set -eux; \
    mkdir -p /opt ${USERBIN_DIR}; \
    chown ${DEV_USER_UID}:${DEV_USER_GID} /opt ${USERBIN_DIR}

ENV PATH="${USERBIN_DIR}:${PATH}"

# dev niceties
RUN --mount=type=cache,id=var-cache-apt,target=/var/cache/apt \
    --mount=type=cache,id=var-lib-apt,target=/var/lib/apt \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends neovim fish; \
    rm -rf /var/lib/apt/lists/*

# set up dev user
RUN set -eux; \
    groupadd --gid ${DEV_USER_GID} dev; \
    useradd --uid ${DEV_USER_UID} --gid ${DEV_USER_GID} -m dev

# move to the dev user's environment now and set up the tools that don't require root
USER dev

WORKDIR /pointcaster

# install qt6 libs
ARG QT_VERSION=6.11.1
ENV QT_INSTALL_DIR=/opt/qt
ENV Qt6_DIR=${QT_INSTALL_DIR}/${QT_VERSION}

RUN set -eux; \
    aqt install-qt \
        --outputdir "${QT_INSTALL_DIR}" \
        linux desktop ${QT_VERSION} linux_gcc_64 \
        -m qtshadertools qtquick3d qttasktree

ENV CMAKE_PREFIX_PATH="${QT_INSTALL_DIR}/${QT_VERSION}/gcc_64"
ENV PATH="${Qt6_DIR}/gcc_64/bin:${PATH}"

# install appimagetool for AppImage deployment
ARG APPIMAGETOOL_VERSION=1.9.1
ARG APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
ENV APPIMAGETOOL_DIR=${USERBIN_DIR}

RUN --mount=type=cache,id=user-downloads,target=/home/dev/.cache/downloads,uid=${DEV_USER_UID},gid=${DEV_USER_GID} \
    set -eux; \
    mkdir -p "${APPIMAGETOOL_DIR}"; \
    cd "${APPIMAGETOOL_DIR}"; \
    test -f "/home/dev/.cache/downloads/appimagetool-x86_64.AppImage" || \
        axel "https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-x86_64.AppImage" \
            -o "/home/dev/.cache/downloads/appimagetool-x86_64.AppImage"; \
    echo "${APPIMAGETOOL_SHA256}  /home/dev/.cache/downloads/appimagetool-x86_64.AppImage" | sha256sum -c -; \
    cp -f "/home/dev/.cache/downloads/appimagetool-x86_64.AppImage" ./appimagetool-x86_64.AppImage; \
    chmod +x appimagetool-x86_64.AppImage; \
    mkdir -p appimagetool && cd appimagetool; \
    ../appimagetool-x86_64.AppImage --appimage-extract

# download and bootstrap vcpkg...
# then run a vcpkg install to bundle the fully-built vcpkg source-based 
# dependencies into the docker image

ENV VCPKG_ROOT=/opt/vcpkg
ENV VCPKG_DEFAULT_BINARY_CACHE=/home/dev/.cache/vcpkg/archives
ENV VCPKG_DOWNLOADS=/opt/vcpkg-cache/downloads

RUN set -eux; mkdir -p ${VCPKG_DEFAULT_BINARY_CACHE} ${VCPKG_DOWNLOADS}
RUN set -eux; mkdir -p /opt/vcpkg-config

RUN --mount=type=cache,id=vcpkg-downloads-cache,target=${VCPKG_DOWNLOADS},uid=${DEV_USER_UID},gid=${DEV_USER_GID} \
    set -eux; \
    git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"; \
    cd "${VCPKG_ROOT}"; \
    git reset --hard $(jq '.["builtin-baseline"]' /opt/vcpkg-config/vcpkg.json -r); \
    ./bootstrap-vcpkg.sh -disableMetrics

ENV PATH="${VCPKG_ROOT}:${PATH}"

ENV VCPKG_KEEP_ENV_VARS="Qt6_DIR;QT_DIR;TBB_DIR;CUDACXX;Thrust_DIR"

COPY vcpkg.json /opt/vcpkg-config/vcpkg.json
COPY triplets /opt/vcpkg-config/triplets
COPY ports /opt/vcpkg-config/ports

# we trigger a vcpkg install using our project's config, overlay triplets and overlay ports 
# in order to populate the system-wide vcpkg binary cache. we can delete the vcpkg_installed there
# and our project's dependencies will be restored into it's own vcpkg_installed from the cache

RUN --mount=type=cache,id=vcpkg-downloads-cache,target=${VCPKG_DOWNLOADS},uid=${DEV_USER_UID},gid=${DEV_USER_GID} \
    set -eux; \
    vcpkg install \
      --x-manifest-root=/opt/vcpkg-config \
      --downloads-root=${VCPKG_DOWNLOADS} \
      --overlay-triplets=/opt/vcpkg-config/triplets \
      --overlay-ports=/opt/vcpkg-config/ports \
      --triplet x64-linux-custom-release \
      --clean-after-build; \
    rm -rf /opt/vcpkg-config/vcpkg_installed