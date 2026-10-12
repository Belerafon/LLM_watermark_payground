# Regenerates version.json — the offline snapshot of the page version shown in
# the header (short commit hash + commit time). Run after committing changes:
#
#   git commit -m "..."
#   powershell -ExecutionPolicy Bypass -File update_version.ps1
#   git add version.json; git commit -m "Update version snapshot"; git push
#
# The page first shows this snapshot, then refreshes it from the GitHub API,
# so a stale snapshot is harmless.

$sha = git log -1 --format="%H"
$short = git log -1 --format="%h"
$time = git log -1 --format="%cI"

if (-not $sha) {
  Write-Error "Not a git repository or no commits yet."
  exit 1
}

$json = @{ sha = $sha.Trim(); commit = $short.Trim(); time = $time.Trim() } | ConvertTo-Json -Compress
$out = Join-Path $PSScriptRoot "version.json"
[IO.File]::WriteAllText($out, $json + "`n", (New-Object System.Text.UTF8Encoding($false)))
Write-Output "version.json -> $short $time"
