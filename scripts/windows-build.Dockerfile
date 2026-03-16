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
      --add Microsoft.VisualStudio.Component.Windows11SDK.26100'; \
    Remove-Item -Force vs_BuildTools.exe

# python+pip, the aqt installer and Qt 6 libs
ARG PythonVersion=3.14.3
ARG QtInstallDirectory="C:\\Qt"
# ARG QtVersion=6.11.0
ARG QtVersion=6.10.3

RUN Invoke-WebRequest -Uri https://www.python.org/ftp/python/$($Env:PythonVersion)/python-$($Env:PythonVersion)-amd64.exe -OutFile python_installer.exe; \
    Start-Process -FilePath .\\python_installer.exe -Wait -ArgumentList \
      '/quiet InstallAllUsers=1 PrependPath=1 Include_test=0'; \
    Remove-Item -Force python_installer.exe
RUN pip install aqtinstall
RUN aqt install-qt \
      --outputdir $($Env:QtInstallDirectory) \
      windows desktop $($Env:QtVersion) win64_msvc2022_64 \
      -m qtshadertools qtquick3d; \
      # -m qtshadertools qtquick3d qttasktree; \
    [Environment]::SetEnvironmentVariable('Qt6_DIR', "$($Env:QtInstallDirectory)\\$($Env:QtVersion)\\lib\\cmake\\Qt6", 'Machine')