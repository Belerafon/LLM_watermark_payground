$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Get-Asset {
    param(
        [string]$Url,
        [string]$Dest,
        [long]$Expected = 0
    )
    $dir = Split-Path -Parent $Dest
    if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    if ($Expected -gt 0 -and (Test-Path $Dest) -and ((Get-Item $Dest).Length -eq $Expected)) {
        Write-Output "SKIP $Dest"
        return
    }
    Write-Output "GET  $Dest"
    $curlArgs = @(
        "-L", "--fail",
        "--retry", "8", "--retry-all-errors", "--retry-delay", "3",
        "-A", "Mozilla/5.0",
        "-o", $Dest,
        $Url
    )
    if (Test-Path $Dest) {
        $curlArgs = @("-C", "-") + $curlArgs
    }
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed ($LASTEXITCODE): $Dest"
    }
    $size = (Get-Item $Dest).Length
    if ($Expected -gt 0 -and $size -ne $Expected) {
        throw "size mismatch for $Dest : got $size expected $Expected"
    }
    Write-Output "OK   $Dest ($size)"
}

$vendor = @(
    @{
        Url = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2/dist/ort-wasm-simd-threaded.jsep.wasm"
        Dest = "vendor\ort-wasm-simd-threaded.jsep.wasm"
        Expected = 21596019
    },
    @{
        Url = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2/dist/ort-wasm-simd-threaded.jsep.mjs"
        Dest = "vendor\ort-wasm-simd-threaded.jsep.mjs"
        Expected = 44484
    },
    @{
        Url = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250409-89f8206ba4/dist/ort.bundle.min.mjs"
        Dest = "vendor\ort.bundle.min.mjs"
        Expected = 398170
    },
    @{
        Url = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250409-89f8206ba4/dist/ort-wasm-simd-threaded.wasm"
        Dest = "vendor\ort-wasm-simd-threaded.wasm"
        Expected = 11133407
    },
    @{
        Url = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250409-89f8206ba4/dist/ort-wasm-simd-threaded.mjs"
        Dest = "vendor\ort-wasm-simd-threaded.mjs"
        Expected = 20856
    }
)

$hf = "https://huggingface.co/onnx-community/Qwen3-4B-ONNX/resolve/main"
$model = "models\onnx-community\Qwen3-4B-ONNX"
$files = @(
    @{ Url = "$hf/config.json"; Dest = "$model\config.json"; Expected = 1780 },
    @{ Url = "$hf/generation_config.json"; Dest = "$model\generation_config.json"; Expected = 219 },
    @{ Url = "$hf/tokenizer_config.json"; Dest = "$model\tokenizer_config.json"; Expected = 9761 },
    @{ Url = "$hf/tokenizer.json"; Dest = "$model\tokenizer.json"; Expected = 9117040 },
    @{ Url = "$hf/special_tokens_map.json"; Dest = "$model\special_tokens_map.json"; Expected = 613 },
    @{ Url = "$hf/added_tokens.json"; Dest = "$model\added_tokens.json"; Expected = 707 },
    @{ Url = "$hf/chat_template.jinja"; Dest = "$model\chat_template.jinja"; Expected = 4168 },
    @{ Url = "$hf/onnx/model_q4f16.onnx"; Dest = "$model\onnx\model_q4f16.onnx"; Expected = 59762833 },
    @{ Url = "$hf/onnx/model_q4f16.onnx_data"; Dest = "$model\onnx\model_q4f16.onnx_data"; Expected = 2096005120 },
    @{ Url = "$hf/onnx/model_q4f16.onnx_data_1"; Dest = "$model\onnx\model_q4f16.onnx_data_1"; Expected = 677150720 }
)

foreach ($item in ($vendor + $files)) {
    Get-Asset -Url $item.Url -Dest $item.Dest -Expected $item.Expected
}
Write-Output "ALL DONE"
