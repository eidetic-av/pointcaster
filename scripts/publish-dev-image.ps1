# pass through script arguments as docker build args (like --no-cache)
param([Parameter(ValueFromRemainingArguments)] [string[]] $DockerBuildArgs)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$Image = 'flowbox:5000/pointcaster-dev'
$Date  = [DateTime]::UtcNow.ToString('yyyy-MM-dd')
$Tag   = "$Date-windows"

Set-Location (Join-Path $PSScriptRoot '..')

Write-Host "==> build ${Image}:${Tag}"
docker build --isolation=hyperv -f scripts/windows.Dockerfile -t "${Image}:${Tag}" `
  --label "org.opencontainers.image.version=$Date" `
  --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" `
  --label "org.opencontainers.image.created=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))" `
  @DockerBuildArgs .

# save to a tarball first so crane can upload from a local file...
# to get around zot deployment behind cloudflare tunnel failing 'docker push' with large objects
$Tar = Join-Path $env:TEMP "pointcaster-dev-$Tag.tar"
try {
  Write-Host '==> save'
  docker save "${Image}:${Tag}" -o $Tar

  Write-Host '==> push'
  crane push --insecure $Tar "${Image}:${Tag}"
} finally {
  Remove-Item $Tar -ErrorAction SilentlyContinue
}

Write-Host "==> done: ${Image}:${Tag}"