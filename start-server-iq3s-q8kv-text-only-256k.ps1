$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$serverExe = Join-Path $scriptDir "llama-cpp\llama-server.exe"
$modelFile = Join-Path $scriptDir "models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"

if (!(Test-Path $serverExe)) { Write-Error "llama-server.exe not found: $serverExe"; exit 1 }
if (!(Test-Path $modelFile)) { Write-Error "Model file not found: $modelFile"; exit 1 }

$existing = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Error "Port 8080 is already in use (PID $($existing.OwningProcess)). Stop the running llama-server first. / 端口 8080 已被占用（PID $($existing.OwningProcess)），请先停止正在运行的实例。"
    exit 1
}

Write-Host "Starting llama-server (IQ3_S-mtp, text-only)..." -ForegroundColor Cyan
Write-Host "  Model: $modelFile" -ForegroundColor Gray
Write-Host "  Context: 256K, KV q8_0 symmetric, Flash Attention, all layers GPU, MTP spec n-max 3, ubatch 512, threads 6" -ForegroundColor Gray
Write-Host "  Sampling (Qwen3.8-27B thinking mode): temp 1.0, top_p 0.95, top_k 20, min_p 0" -ForegroundColor Gray

& $serverExe `
  -m $modelFile `
  -c 262144 --flash-attn on `
  --cache-type-k q8_0 --cache-type-v q8_0 `
  --alias qwen3.8-27b --no-mmproj --parallel 1 `
  -ngl all -t 6 -ub 512 `
  --spec-type draft-mtp `
  --spec-draft-n-max 3 `
  --spec-draft-poll 1 `
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 `
  --presence-penalty 0.0 --repeat-penalty 1.0 `
  --host 0.0.0.0 --port 8080 -cram 10240

