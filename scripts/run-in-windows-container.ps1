# Run a command inside the pointcaster-dev windows container, e.g.
#
#   scripts/run-in-windows-container.ps1 cmake --preset windows-release
#

$ErrorActionPreference = 'Stop'

$command = @($args)
if ($command.Count -eq 0) {
    throw "usage: run-in-windows-container.ps1 <command> [args...]"
}

function Get-EnvOrDefault([string] $Name, [string] $Fallback) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Fallback }
    return $value
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$image    = Get-EnvOrDefault 'POINTCASTER_DEV_IMAGE'    'cr.pointcaster.net/pointcaster-dev:2026-07-26-windows'
$volume   = Get-EnvOrDefault 'POINTCASTER_BUILD_VOLUME' 'pointcaster-build-cache'
$cpus     = Get-EnvOrDefault 'POINTCASTER_BUILD_CPUS'   '22'
$memory   = Get-EnvOrDefault 'POINTCASTER_BUILD_MEMORY' '14g'

$dockerArgs = [System.Collections.Generic.List[string]]@(
    'run', '--rm',
    '--isolation=hyperv',
    '--cpus', $cpus,
    '--memory', $memory,
    '-v', "${repoRoot}:C:\pointcaster",
    '-v', "${volume}:C:\pointcaster\build",
    '-w', 'C:\pointcaster'
)

foreach ($name in 'DEPLOY_BRANCH', 'B2_APPLICATION_KEY_ID', 'B2_APPLICATION_KEY') {
    if (-not [string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($name))) {
        $dockerArgs.AddRange([string[]] @('-e', $name))
    }
}

$dockerArgs.Add($image)
$dockerArgs.AddRange([string[]] $command)

Write-Host "==> $image : $($command -join ' ')"
& docker @dockerArgs

if ($LASTEXITCODE -ne 0) {
    throw "container command failed (exit $LASTEXITCODE): $($command -join ' ')"
}
