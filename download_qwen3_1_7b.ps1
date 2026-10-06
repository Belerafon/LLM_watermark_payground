$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Get-Asset {
    param([string]$Url, [string]$Dest)
    $dir = Split-Path -Parent $Dest
    if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    if (Test-Path $Dest) {
        Write-Output "SKIP $Dest"
        return
    }
    $partial = "$Dest.part"
    Write-Output "GET  $Dest"
    $curlArgs = @("-L", "--fail", "--retry", "8", "--retry-all-errors", "--retry-delay", "3", "-A", "Mozilla/5.0", "-o", $partial, $Url)
    if (Test-Path $partial) { $curlArgs = @("-C", "-") + $curlArgs }
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE): $Dest" }
    Move-Item -LiteralPath $partial -Destination $Dest -Force
    Write-Output "OK   $Dest ($((Get-Item $Dest).Length))"
}

$hf = "https://huggingface.co/onnx-community/Qwen3-1.7B-ONNX/resolve/main"
$model = "models\onnx-community\Qwen3-1.7B-ONNX"
$files = @(
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "onnx/model_q4f16.onnx"
)

foreach ($file in $files) {
    $dest = Join-Path $model ($file -replace '/', '\')
    Get-Asset -Url "$hf/$file" -Dest $dest
}

Write-Output "QWEN3-1.7B Q4F16 DONE"
