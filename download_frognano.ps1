$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Get-Asset {
    param([string]$Url, [string]$Dest, [long]$Expected = 0)
    $dir = Split-Path -Parent $Dest
    if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    if ($Expected -gt 0 -and (Test-Path $Dest) -and ((Get-Item $Dest).Length -eq $Expected)) {
        Write-Output "SKIP $Dest"
        return
    }
    Write-Output "GET  $Dest"
    $curlArgs = @("-L", "--fail", "--retry", "8", "--retry-all-errors", "--retry-delay", "3", "-A", "Mozilla/5.0", "-o", $Dest, $Url)
    if (Test-Path $Dest) { $curlArgs = @("-C", "-") + $curlArgs }
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE): $Dest" }
    $size = (Get-Item $Dest).Length
    if ($Expected -gt 0 -and $size -ne $Expected) { throw "size mismatch for $Dest : got $size expected $Expected" }
    Write-Output "OK   $Dest ($size)"
}

$hf = "https://huggingface.co/RASMUS/FrogNano-4B-2609-ONNX/resolve/main"
$model = "models\RASMUS\FrogNano-4B-2609-ONNX"
$files = @(
    @{ Url = "$hf/config.json"; Dest = "$model\config.json"; Expected = 0 },
    @{ Url = "$hf/generation_config.json"; Dest = "$model\generation_config.json"; Expected = 0 },
    @{ Url = "$hf/tokenizer_config.json"; Dest = "$model\tokenizer_config.json"; Expected = 0 },
    @{ Url = "$hf/tokenizer.json"; Dest = "$model\tokenizer.json"; Expected = 12807982 },
    @{ Url = "$hf/chat_template.jinja"; Dest = "$model\chat_template.jinja"; Expected = 0 },
    @{ Url = "$hf/onnx/model_q4f16.onnx"; Dest = "$model\onnx\model_q4f16.onnx"; Expected = 655272 },
    @{ Url = "$hf/onnx/model_q4f16.onnx_data"; Dest = "$model\onnx\model_q4f16.onnx_data"; Expected = 1990742016 },
    @{ Url = "$hf/onnx/model_q4f16.onnx_data_1"; Dest = "$model\onnx\model_q4f16.onnx_data_1"; Expected = 444579840 }
)

foreach ($item in $files) {
    Get-Asset -Url $item.Url -Dest $item.Dest -Expected $item.Expected
}
Write-Output "FROGNANO DONE"
