FROM mcr.microsoft.com/windows/server:ltsc2022-KB5078766-amd64

SHELL ["powershell", "-ExecutionPolicy", "Bypass", "-Command"]

# Enable long paths and dev mode
RUN Set-ItemProperty -Path HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem -Name LongPathsEnabled -Value 1
RUN Set-ItemProperty -Path HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock -Name AllowDevelopmentWithoutDevLicense -Value 1 -Type DWord

# C++ and Windows SDK components
RUN Invoke-WebRequest -Uri https://aka.ms/vs/17/release/vs_BuildTools.exe -OutFile vs_BuildTools.exe; \
    Start-Process -FilePath .\\vs_BuildTools.exe -Wait -ArgumentList \
      '--quiet --norestart \
      --add Microsoft.VisualStudio.Component.VC.CoreBuildTools \
      --add Microsoft.VisualStudio.Component.VC.Redist.14.Latest \
      --add Microsoft.VisualStudio.Component.Windows10SDK \
      --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core \
      --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64 \
      --add Microsoft.VisualStudio.Component.VC.CMake.Project \
      --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
      --add Microsoft.VisualStudio.Component.Vcpkg \
      --add Microsoft.VisualStudio.Component.Windows11SDK.26100 \
      --add Microsoft.VisualStudio.Component.VC.ATL \
      --add Microsoft.VisualStudio.Component.VC.ATLMFC'; \
    Remove-Item -Force vs_BuildTools.exe

# python+pip, the aqt installer and Qt 6 libs
ARG PythonVersion=3.14.3
RUN Invoke-WebRequest -Uri https://www.python.org/ftp/python/$Env:PythonVersion/python-$Env:PythonVersion-amd64.exe -OutFile python_installer.exe; \
    Start-Process -FilePath .\\python_installer.exe -Wait -ArgumentList \
      '/quiet InstallAllUsers=1 PrependPath=1 Include_test=0'; \
    Remove-Item -Force python_installer.exe

ARG QtInstallDirectory="C:\\Qt"
ARG QtVersion=6.11.0
ENV Qt6_DIR="C:\\Qt\\6.11.0\\msvc2022_64\\lib\\cmake\\Qt6"
ENV QT_DIR="C:\\Qt\\6.11.0\\msvc2022_64\\lib\\cmake\\Qt6"

# ARG AqtInstallVersion=3.3
# RUN pip install "aqtinstall==$Env:AqtInstallVersion"
# temporary workaround for 6.11.0
RUN pip install git+https://github.com/miurahr/aqtinstall.git@refs/pull/1000/head
RUN aqt install-qt \
      --outputdir "$Env:QtInstallDirectory" \
      windows desktop "$Env:QtVersion" win64_msvc2022_64 \
      -m qtshadertools qtquick3d qttasktree

# oneTBB
ARG TbbVersion=2022.3.0
ARG TbbInstallDir="C:\\TBB"
ENV TBB_DIR="C:\\TBB\\oneapi-tbb-2022.3.0"
RUN mkdir "$Env:TbbInstallDir"; \
      Invoke-WebRequest "https://github.com/uxlfoundation/oneTBB/releases/download/v$Env:TbbVersion/oneapi-tbb-$Env:TbbVersion-win.zip" \
            -OutFile "$Env:TbbInstallDir\\oneapi-tbb-$Env:TbbVersion-win.zip"; \
      Expand-Archive -Path "$Env:TbbInstallDir\\oneapi-tbb-$Env:TbbVersion-win.zip" \
            -DestinationPath "$Env:TbbInstallDir"; \
      rm "$Env:TbbInstallDir\\oneapi-tbb-$Env:TbbVersion-win.zip"

# jinja is used for reflection / templated code generation scripts
ARG Jinja2Version=3.1.6
RUN pip install "jinja2==$Env:Jinja2Version"

# entry point to the docker container is our visual studio dev shell
# so env with build tools is properly configured

ENV VsDevShell="C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\Launch-VsDevShell.ps1"
ENTRYPOINT [ "powershell", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command", "& $Env:VsDevShell -Arch amd64 -HostArch amd64;& " ]
