FROM docker.io/zhongruoyu/gcc-ports:15.2-bookworm

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV ARCH=x86_64

ARG QT_VERSION=6.11.0

ARG APPIMAGETOOL_VERSION=1.9.1
ARG APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"

ARG VCPKG_COMMIT=1dce60b00170a48594cdb9adbcecf03a26066fec

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
        libtinfo6 libzstd1 zlib1g; \
    rm -rf /var/lib/apt/lists/*

# download and install cmake
ARG CMAKE_VERSION=4.2.1
RUN set -eux; \
    BASE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}"; \
    curl -fsSLO "${BASE_URL}/cmake-${CMAKE_VERSION}-SHA-256.txt"; \
    curl -fsSLO "${BASE_URL}/cmake-${CMAKE_VERSION}-linux-x86_64.sh"; \
    grep "cmake-${CMAKE_VERSION}-linux-x86_64.sh" "cmake-${CMAKE_VERSION}-SHA-256.txt" > cmake.sha256; \
    sha256sum -c cmake.sha256; \
    sh "cmake-${CMAKE_VERSION}-linux-x86_64.sh" --skip-license --prefix=/usr/local; \
    rm -f cmake-${CMAKE_VERSION}-linux-x86_64.sh cmake-${CMAKE_VERSION}-SHA-256.txt cmake.sha256

# add qt installer (aqt)
RUN --mount=type=cache,id=root-cache-pip,target=/root/.cache/pip \
    set -eux; \
    python3 -m pip install --no-input aqtinstall --break-system-packages

# set up dev user
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN set -eux; \
    groupadd --gid "${USER_GID}" "${USERNAME}"; \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}"

# set up extra directories for dependency locations manipulated by the dev user
ENV USERBIN_DIR=/opt/bin
ENV VCPKG_ROOT=/opt/vcpkg
ENV QT_INSTALL_DIR=/opt/qt
ENV Qt6_DIR=${QT_INSTALL_DIR}/${QT_VERSION}
ENV APPIMAGETOOL_DIR=${USERBIN_DIR}

ENV PATH="${USERBIN_DIR}:${VCPKG_ROOT}:${Qt6_DIR}/gcc_64/bin:${PATH}"

RUN set -eux; \
    mkdir -p /opt ${USERBIN_DIR}; \
    chown "${USER_UID}:${USER_GID}" /opt ${USERBIN_DIR}

# dev niceties
RUN --mount=type=cache,id=var-cache-apt,target=/var/cache/apt \
    --mount=type=cache,id=var-lib-apt,target=/var/lib/apt \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends neovim fish; \
    rm -rf /var/lib/apt/lists/*

# install clangd + clang-format + clang-tidy (LLVM 21)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates wget gnupg; \
    rm -rf /var/lib/apt/lists/*; \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor > /usr/share/keyrings/llvm-snapshot.gpg; \
    printf 'deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-21 main\n' > /etc/apt/sources.list.d/llvm21.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends clangd-21 clang-tools-21; \
    rm -rf /var/lib/apt/lists/*

# optionally install vscode for in-container ide setup
ARG INSTALL_VSCODE=0
ARG VSCODE_VERSION=1.108.0
ARG VSCODE_BUILD_TIMESTAMP=1767881962
ARG VSCODE_SHA256=1722e0cf7b72a6806c9dc24b755067635d4831ff6f927f1a93642052cc4a364f

RUN --mount=type=cache,id=var-cache-downloads,target=/var/cache/downloads \
    set -eux; \
    if [ "${INSTALL_VSCODE}" = "1" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 libgtk-3-0 \
            libnspr4 libnss3 libxcomposite1 libxdamage1 libxkbfile1 libxrandr2 xdg-utils; \
        VSCODE_DEB="code_${VSCODE_VERSION}-${VSCODE_BUILD_TIMESTAMP}_amd64.deb"; \
        if [ ! -f "/var/cache/downloads/${VSCODE_DEB}" ]; then \
            axel -n 8 -o "/var/cache/downloads/${VSCODE_DEB}" \
                "https://update.code.visualstudio.com/${VSCODE_VERSION}/linux-deb-x64/stable"; \
        fi; \
        echo "${VSCODE_SHA256}  /var/cache/downloads/${VSCODE_DEB}" | sha256sum -c -; \
        export DEBIAN_FRONTEND=noninteractive; \
        echo "code code/add-microsoft-repo boolean true" | debconf-set-selections; \
        apt-get install -y --no-install-recommends "/var/cache/downloads/${VSCODE_DEB}"; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# move to the dev user's environment now and set it up
USER ${USERNAME}

WORKDIR /pointcaster

# install qt6 libs
RUN set -eux; \
    aqt install-qt \
        --outputdir "${QT_INSTALL_DIR}" \
        linux desktop ${QT_VERSION} linux_gcc_64 \
        -m qtshadertools qtquick3d

ENV CMAKE_PREFIX_PATH="${QT_INSTALL_DIR}/${QT_VERSION}/gcc_64"

# download and bootstrap vcpkg into the user's home dir
ENV VCPKG_DEFAULT_BINARY_CACHE=/home/${USERNAME}/.cache/vcpkg/archives
ENV VCPKG_DOWNLOADS=${VCPKG_ROOT}-cache/downloads
ENV VCPKG_BUILDTREES=${VCPKG_ROOT}-cache/buildtrees

RUN --mount=type=cache,id=vcpkg-downloads,target=${VCPKG_DOWNLOADS},uid=${USER_UID},gid=${USER_GID} \
    --mount=type=cache,id=vcpkg-buildtrees,target=${VCPKG_BUILDTREES},uid=${USER_UID},gid=${USER_GID} \
    set -eux; \
    mkdir -p ${VCPKG_DEFAULT_BINARY_CACHE}; \
    git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"; \
    cd "${VCPKG_ROOT}"; \
    git reset --hard "${VCPKG_COMMIT}"; \
    ./bootstrap-vcpkg.sh -disableMetrics

# install appimagetool for AppImage deployment
RUN --mount=type=cache,id=user-downloads,target=/home/${USERNAME}/.cache/downloads,uid=${USER_UID},gid=${USER_GID} \
    set -eux; \
    mkdir -p "${APPIMAGETOOL_DIR}"; \
    cd "${APPIMAGETOOL_DIR}"; \
    test -f "/home/${USERNAME}/.cache/downloads/appimagetool-x86_64.AppImage" || \
        axel "https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-x86_64.AppImage" \
            -o "/home/${USERNAME}/.cache/downloads/appimagetool-x86_64.AppImage"; \
    echo "${APPIMAGETOOL_SHA256}  /home/${USERNAME}/.cache/downloads/appimagetool-x86_64.AppImage" | sha256sum -c -; \
    cp -f "/home/${USERNAME}/.cache/downloads/appimagetool-x86_64.AppImage" ./appimagetool-x86_64.AppImage; \
    chmod +x appimagetool-x86_64.AppImage; \
    mkdir -p appimagetool && cd appimagetool; \
    ../appimagetool-x86_64.AppImage --appimage-extract
