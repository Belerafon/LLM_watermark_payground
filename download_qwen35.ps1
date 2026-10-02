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

$ort = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist"
$tf = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist"
$vendor = @(
    @{ Url = "$tf/transformers.min.js"; Dest = "vendor4\transformers.min.js"; Expected = 581935 },
    @{ Url = "$ort/ort-wasm-simd-threaded.jsep.mjs"; Dest = "vendor4\ort-wasm-simd-threaded.jsep.mjs"; Expected = 46851 },
    @{ Url = "$ort/ort-wasm-simd-threaded.jsep.wasm"; Dest = "vendor4\ort-wasm-simd-threaded.jsep.wasm"; Expected = 28352885 },
    @{ Url = "$ort/ort-wasm-simd-threaded.asyncify.mjs"; Dest = "vendor4\ort-wasm-simd-threaded.asyncify.mjs"; Expected = 53057 },
    @{ Url = "$ort/ort-wasm-simd-threaded.asyncify.wasm"; Dest = "vendor4\ort-wasm-simd-threaded.asyncify.wasm"; Expected = 26861777 },
    @{ Url = "$ort/ort.webgpu.bundle.min.mjs"; Dest = "vendor4\ort.webgpu.bundle.min.mjs"; Expected = 117929 },
    @{ Url = "$ort/ort.bundle.min.mjs"; Dest = "vendor4\ort.bundle.min.mjs"; Expected = 413592 }
)

$hf = "https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT/resolve/main"
$model = "models\onnx-community\Qwen3.5-2B-ONNX-OPT"
$files = @(
    @{ Url = "$hf/config.json"; Dest = "$model\config.json"; Expected = 0 },
    @{ Url = "$hf/generation_config.json"; Dest = "$model\generation_config.json"; Expected = 0 },
    @{ Url = "$hf/tokenizer_config.json"; Dest = "$model\tokenizer_config.json"; Expected = 0 },
    @{ Url = "$hf/tokenizer.json"; Dest = "$model\tokenizer.json"; Expected = 0 },
    @{ Url = "$hf/preprocessor_config.json"; Dest = "$model\preprocessor_config.json"; Expected = 0 },
    @{ Url = "$hf/processor_config.json"; Dest = "$model\processor_config.json"; Expected = 0 },
    @{ Url = "$hf/chat_template.jinja"; Dest = "$model\chat_template.jinja"; Expected = 0 },
    @{ Url = "$hf/onnx/decoder_model_merged_q4f16.onnx"; Dest = "$model\onnx\decoder_model_merged_q4f16.onnx"; Expected = 707377 },
    @{ Url = "$hf/onnx/decoder_model_merged_q4f16.onnx_data"; Dest = "$model\onnx\decoder_model_merged_q4f16.onnx_data"; Expected = 1088892928 },
    @{ Url = "$hf/onnx/embed_tokens_q4f16.onnx"; Dest = "$model\onnx\embed_tokens_q4f16.onnx"; Expected = 1064 },
    @{ Url = "$hf/onnx/embed_tokens_q4f16.onnx_data"; Dest = "$model\onnx\embed_tokens_q4f16.onnx_data"; Expected = 294010880 },
    @{ Url = "$hf/onnx/vision_encoder_q4f16.onnx"; Dest = "$model\onnx\vision_encoder_q4f16.onnx"; Expected = 394142 },
    @{ Url = "$hf/onnx/vision_encoder_q4f16.onnx_data"; Dest = "$model\onnx\vision_encoder_q4f16.onnx_data"; Expected = 196945920 }
)

foreach ($item in ($vendor + $files)) {
    Get-Asset -Url $item.Url -Dest $item.Dest -Expected $item.Expected
}
Write-Output "QWEN35 DONE"
