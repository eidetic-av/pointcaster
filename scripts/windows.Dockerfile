FROM mcr.microsoft.com/windows/server:ltsc2022-KB5078766-amd64

SHELL ["powershell", "-ExecutionPolicy", "Bypass", "-Command"]

# Enable long paths and dev mode
RUN Set-ItemProperty -Path HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem -Name LongPathsEnabled -Value 1
RUN Set-ItemProperty -Path HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock -Name AllowDevelopmentWithoutDevLicense -Value 1 -Type DWord

# C++ and Windows SDK components
# fixed-version bootstrapper for VS 2022 17.14.36
# and matching toolset & sdk versions also pinned
ARG VsBuildToolsVersion="17.14.36"
ARG VsBuildToolsUrl="https://download.visualstudio.microsoft.com/download/pr/12aa1305-dd17-4f26-8429-d072cda64c80/5ae95bb02bb3442441a8d891e5bb1d2975445e2e3ee16ada5bc7bd17227f1dd7/vs_BuildTools.exe"
ARG VsBuildToolsSha256="5AE95BB02BB3442441A8D891E5BB1D2975445E2E3EE16ADA5BC7BD17227F1DD7"
RUN Invoke-WebRequest -Uri $Env:VsBuildToolsUrl -OutFile vs_BuildTools.exe; \
    $ActualHash = (Get-FileHash vs_BuildTools.exe -Algorithm SHA256).Hash; \
    if ($ActualHash -ne $Env:VsBuildToolsSha256) { throw \"vs_BuildTools.exe checksum mismatch: expected $Env:VsBuildToolsSha256 but got $ActualHash\" }; \
    Start-Process -FilePath .\\vs_BuildTools.exe -Wait -ArgumentList \
      '--quiet --norestart --nocache \
      --add Microsoft.VisualStudio.Component.VC.CoreBuildTools \
      --add Microsoft.VisualStudio.Component.VC.Redist.14.Latest \
      --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core \
      --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64 \
      --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
      --add Microsoft.VisualStudio.Component.Windows11SDK.26100 \
      --add Microsoft.VisualStudio.Component.VC.14.44.17.14.ATL \
      --add Microsoft.VisualStudio.Component.VC.14.44.17.14.MFC'; \
    Remove-Item -Force vs_BuildTools.exe

ENV VsDevShell="C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\Launch-VsDevShell.ps1"

# install chocolatey for package management
RUN [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;\
	iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Git
ARG GitVersion=2.55.0.3
RUN choco install -y git --version $Env:GitVersion

# python+pip, the aqt installer and Qt 6 libs
ARG PythonVersion=3.14.6
RUN choco install -y python --version $Env:PythonVersion --installargs 'InstallAllUsers=1 PrependPath=1 Include_test=0'

ARG QtInstallDirectory="C:\\Qt"
ARG QtVersion=6.11.1
ENV Qt6_DIR="C:\\Qt\\6.11.1\\msvc2022_64\\lib\\cmake\\Qt6"
ENV QT_DIR="C:\\Qt\\6.11.1\\msvc2022_64\\lib\\cmake\\Qt6"

# ARG AqtInstallVersion=3.3
# RUN pip install "aqtinstall==$Env:AqtInstallVersion"
# temporary workaround to properly install 6.11 until new aqt release with fix
RUN pip install --no-cache-dir git+https://github.com/miurahr/aqtinstall.git@f383b3c7d9658e881bd8ce8810057393bd934ee1
RUN aqt install-qt \
      --outputdir "$Env:QtInstallDirectory" \
      windows desktop "$Env:QtVersion" win64_msvc2022_64 \
      -m qtshadertools qtquick3d qttasktree

# oneTBB
ARG TbbVersion=2023.1.0
ARG TbbInstallDir="C:\\TBB"
ARG TbbSha256="CF6EE0C600FCB5C3A9B65E3E6E4781669D06F1BB1E37970D145FCDE08EED8DA9"
ENV TBB_DIR="C:\\TBB\\oneapi-tbb-2023.1.0"
RUN mkdir "$Env:TbbInstallDir"; \
      $TbbZip = \"$Env:TbbInstallDir\oneapi-tbb-$Env:TbbVersion-win.zip\"; \
      Invoke-WebRequest \"https://github.com/uxlfoundation/oneTBB/releases/download/v$Env:TbbVersion/oneapi-tbb-$Env:TbbVersion-win.zip\" -OutFile $TbbZip; \
      $ActualHash = (Get-FileHash $TbbZip -Algorithm SHA256).Hash; \
      if ($ActualHash -ne $Env:TbbSha256) { throw \"oneTBB checksum mismatch: expected $Env:TbbSha256 but got $ActualHash\" }; \
      Expand-Archive -Path $TbbZip -DestinationPath \"$Env:TbbInstallDir\"; \
      rm $TbbZip

# jinja is used for reflection / templated code generation scripts
ARG Jinja2Version=3.1.6
RUN pip install "jinja2==$Env:Jinja2Version"

# NVIDIA CUDA development packages
ARG CudaVersion=12.9.1.576
RUN choco install -y cuda --version $Env:CudaVersion

ENV CUDAToolkit_ROOT="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
ENV CUDACXX="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin\\nvcc.exe"
ENV Thrust_DIR="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\lib\\cmake\\thrust"

# Build tools from choco

ARG CMakeVersion=4.4.0
RUN choco install -y cmake --version $Env:CMakeVersion --installargs 'ADD_CMAKE_TO_PATH=System'

ARG NinjaVersion=1.13.2
RUN choco install -y ninja --version $Env:NinjaVersion

# Build source-based project dependencies with vcpkg
COPY vcpkg.json C:\\vcpkg-config\\
COPY triplets C:\\vcpkg-config\\triplets
COPY ports C:\\vcpkg-config\\ports

# bootstrap our own vcpkg checkout pinned to the manifest's builtin-baseline commit
# - this ensures vcpkg tooling is at a known version
ENV VCPKG_ROOT="C:\\vcpkg"
RUN git clone https://github.com/microsoft/vcpkg.git $Env:VCPKG_ROOT; \
    cd $Env:VCPKG_ROOT; \
    $Baseline = (Get-Content C:\\vcpkg-config\\vcpkg.json | ConvertFrom-Json).'builtin-baseline'; \
    git reset --hard $Baseline; \
    & .\\bootstrap-vcpkg.bat -disableMetrics

ENV VCPKG_KEEP_ENV_VARS="Qt6_DIR;QT_DIR;TBB_DIR;CUDAToolkit_ROOT;CUDACXX;Thrust_DIR"

# activate the VS dev shell and install third-party source-based
# project dependencies as precompiled libs in this image using vcpkg

RUN & \"$Env:VsDevShell\" -Arch amd64 -HostArch amd64; \
    & \"$Env:VCPKG_ROOT\vcpkg.exe\" install \
      --x-manifest-root=C:\vcpkg-config \
      --overlay-triplets=C:\vcpkg-config\triplets \
      --overlay-ports=C:\vcpkg-config\ports \
      --triplet x64-windows-static-md-custom-release \
      --clean-after-build

WORKDIR C:\\pointcaster

# entry point to the docker container is our visual studio dev shell
ENTRYPOINT [ "powershell", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command", "& $Env:VsDevShell -Arch amd64 -HostArch amd64; & " ]
