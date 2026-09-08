# Copy the built native plugin into the Unity package.
#
#   ./scripts/stage-unity-package.ps1 -InstallDir build/pointreceiver-windows-release/install
#
# Plugins/<Platform>/<Arch> is gitignored here but its .meta files are committed,
# and Unity deletes a .meta file whose asset is missing - so the binaries have to
# be in place before the package is opened in an editor.

param(
    [string] $InstallDir = 'build/pointreceiver-windows-release/install',
    [string] $Platform = 'Windows',
    [string] $Arch = 'x86_64'
)

$ErrorActionPreference = 'Stop'

$repo_root = Resolve-Path (Join-Path $PSScriptRoot '..')

# only the runtime libraries: the import library under lib/ and the C headers
# under include/ are for native consumers, Unity has no use for them.
# recursive because windows installs its dlls at the install root (the root
# CMakeLists sets CMAKE_INSTALL_BINDIR "." on WIN32) while android puts
# libpointreceiver.so under lib/
$binaries = Get-ChildItem (Join-Path $repo_root $InstallDir) -File -Recurse |
    Where-Object Extension -in '.dll', '.so'
if (-not $binaries) { throw "no runtime libraries in $InstallDir" }

$plugin_dir = Join-Path $repo_root "src/pointreceiver/unity/Plugins/$Platform/$Arch"
New-Item -ItemType Directory -Path $plugin_dir -Force | Out-Null
$binaries | Copy-Item -Destination $plugin_dir -Force

Write-Host "==> staged $($binaries.Count) libraries into Plugins/$Platform/$Arch"
