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

$hf = "https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX/resolve/main"
$model = "models\onnx-community\gemma-4-E4B-it-ONNX"
$files = @(
    @{ Url = "$hf/config.json"; Dest = "$model\config.json"; Expected = 0 },
    @{ Url = "$hf/generation_config.json"; Dest = "$model\generation_config.json"; Expected = 0 },
    @{ Url = "$hf/tokenizer_config.json"; Dest = "$model\tokenizer_config.json"; Expected = 0 },
    @{ Url = "$hf/tokenizer.json"; Dest = "$model\tokenizer.json"; Expected = 0 },
    @{ Url = "$hf/preprocessor_config.json"; Dest = "$model\preprocessor_config.json"; Expected = 0 },
    @{ Url = "$hf/processor_config.json"; Dest = "$model\processor_config.json"; Expected = 0 },
    @{ Url = "$hf/chat_template.jinja"; Dest = "$model\chat_template.jinja"; Expected = 0 },
    @{ Url = "$hf/onnx/decoder_model_merged_q4f16.onnx"; Dest = "$model\onnx\decoder_model_merged_q4f16.onnx"; Expected = 850610 },
    @{ Url = "$hf/onnx/decoder_model_merged_q4f16.onnx_data"; Dest = "$model\onnx\decoder_model_merged_q4f16.onnx_data"; Expected = 2074847232 },
    @{ Url = "$hf/onnx/decoder_model_merged_q4f16.onnx_data_1"; Dest = "$model\onnx\decoder_model_merged_q4f16.onnx_data_1"; Expected = 812318720 },
    @{ Url = "$hf/onnx/embed_tokens_q4f16.onnx"; Dest = "$model\onnx\embed_tokens_q4f16.onnx"; Expected = 5619 },
    @{ Url = "$hf/onnx/embed_tokens_q4f16.onnx_data"; Dest = "$model\onnx\embed_tokens_q4f16.onnx_data"; Expected = 2017460224 },
    @{ Url = "$hf/onnx/vision_encoder_q4f16.onnx"; Dest = "$model\onnx\vision_encoder_q4f16.onnx"; Expected = 189126 },
    @{ Url = "$hf/onnx/vision_encoder_q4f16.onnx_data"; Dest = "$model\onnx\vision_encoder_q4f16.onnx_data"; Expected = 100762304 },
    @{ Url = "$hf/onnx/audio_encoder_q4f16.onnx"; Dest = "$model\onnx\audio_encoder_q4f16.onnx"; Expected = 260446 },
    @{ Url = "$hf/onnx/audio_encoder_q4f16.onnx_data"; Dest = "$model\onnx\audio_encoder_q4f16.onnx_data"; Expected = 172167424 }
)

foreach ($item in $files) {
    Get-Asset -Url $item.Url -Dest $item.Dest -Expected $item.Expected
}
Write-Output "GEMMA4 DONE"
