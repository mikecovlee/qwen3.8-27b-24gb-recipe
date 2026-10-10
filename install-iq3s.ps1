#Requires -Version 5.1
<#
.SYNOPSIS
    One-click installer for the IQ3_S-mtp recipe (Qwen3.8-27B) on a 24GB GPU.
    Qwen3.8-27B IQ3_S-mtp 配方单卡一键安装脚本。

.DESCRIPTION
    Downloads llama.cpp (b11491, CUDA 13.4) and the ISTA-DASLab Qwen3.8-27B-GSQ-RCO
    IQ3_S-mtp GGUF model, plus the matching vision projector (mmproj) unless -TextOnly.
    下载 llama.cpp 引擎（b11491, CUDA 13.4）与 ISTA-DASLab IQ3_S-mtp 模型；默认同时下载视觉编码器
    （mmproj），使用 -TextOnly 可跳过。

.PARAMETER SkipModel
    Skip downloading the model (engine only). 跳过模型下载，仅安装引擎。

.PARAMETER TextOnly
    Do not download the vision projector (mmproj); text-only deployment.
    不下载视觉编码器（mmproj），仅文本部署。

.PARAMETER Proxy
    Proxy URL for downloads, e.g. http://127.0.0.1:7890. 下载代理地址。

.PARAMETER Force
    Re-download existing files. 已存在文件重新下载。

.PARAMETER LlamaCppUrl
    Override llama.cpp release zip URL. 覆盖引擎下载地址。

.PARAMETER ModelUrl
    Override model GGUF URL. 覆盖模型下载地址。

.PARAMETER MmprojUrl
    Override vision projector URL. 覆盖 mmproj 下载地址。

.EXAMPLE
    .\install-iq3s.ps1
    .\install-iq3s.ps1 -TextOnly
    .\install-iq3s.ps1 -Proxy http://127.0.0.1:7890
#>
param(
    [switch]$SkipModel,
    [switch]$TextOnly,
    [string]$Proxy = "",
    [switch]$Force,
    [string]$LlamaCppUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b11491/llama-b11491-bin-win-cuda-13.4-x64.zip",
    [string]$ModelUrl = "https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
    [string]$MmprojUrl = "https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/mmproj-Qwen3.8-27B-BF16.gguf"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$llamaCppDir = Join-Path $scriptDir "llama-cpp"
$modelsDir = Join-Path $scriptDir "models"
$llamaServerExe = Join-Path $llamaCppDir "llama-server.exe"
$modelFile = Join-Path $modelsDir "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
$mmprojFile = Join-Path $modelsDir "mmproj-Qwen3.8-27B-BF16.gguf"

# Known-good sizes (fallback if Content-Length cannot be read).
$modelSize = 12120016960
$mmprojSize = 931146528

$script:stepCount = 0
function Write-Step($msgEn, $msgZh) { $script:stepCount++; Write-Host "[$script:stepCount/5] $msgEn / $msgZh" -ForegroundColor Cyan }
function Write-Info($msgEn, $msgZh) { Write-Host "  - $msgEn / $msgZh" -ForegroundColor Gray }

# Resumable download that survives curl dying mid-file through a proxy (exit code 0).
function Invoke-ResumableDownload([string]$Url, [string]$OutFile, [long]$ExpectedSize) {
    $proxyArg = @(); if ($Proxy) { $proxyArg = @("-x", $Proxy) }
    $target = $ExpectedSize
    if ($target -le 0) {
        try {
            $cl = (& curl.exe @proxyArg -sIL $Url | Select-String -Pattern '^content-length:\s*(\d+)' | Select-Object -Last 1)
            if ($cl) { $target = [long]$cl.Matches[0].Groups[1].Value }
        } catch { }
    }
    $last = -1
    while ($true) {
        $size = if (Test-Path $OutFile) { (Get-Item $OutFile).Length } else { 0 }
        if ($target -gt 0 -and $size -ge $target) { break }
        if ($target -le 0 -and $last -eq $size -and $size -gt 0) { break }
        $last = $size
        & curl.exe @proxyArg -L -C - --http1.1 --retry 3 --retry-all-errors -o $OutFile $Url
        $newSize = if (Test-Path $OutFile) { (Get-Item $OutFile).Length } else { 0 }
        if ($target -gt 0 -and $newSize -ge $target) { break }
        if ($target -le 0 -and $newSize -gt 0 -and $LASTEXITCODE -eq 0) { break }
    }
    return (Get-Item $OutFile).Length
}

# ---------- Step 1: environment checks ----------
Write-Step "Checking environment..." "检查环境..."
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Error "nvidia-smi not found. This deployment requires an NVIDIA GPU with at least 24GB VRAM. / 未找到 nvidia-smi，本部署需要至少 24GB 显存的 NVIDIA 显卡。"
}
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    Write-Error "curl.exe not found (required for downloads). / 未找到 curl.exe（下载依赖）。"
}

$gpuLine = & nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader 2>$null | Select-Object -First 1
if ($gpuLine) {
    $gpuParts = $gpuLine -split ","
    $totalMB = [int]($gpuParts[0] -replace "[^0-9]", "")
    Write-Info "GPU detected: $($gpuParts[0].Trim()) total / $($gpuParts[1].Trim()) free" "检测到 GPU：共 $($gpuParts[0].Trim()) / 空闲 $($gpuParts[1].Trim())"
    if ($totalMB -lt 24000) {
        Write-Host "  [WARN] VRAM is below 24GB; this model may not fit. / 警告：显存小于 24GB，可能无法加载本模型。" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [WARN] Could not query GPU memory. / 警告：无法查询显存信息。" -ForegroundColor Yellow
}

$freeGB = (Get-PSDrive (Split-Path $scriptDir -Qualifier).TrimEnd(':')).Free / 1GB
$needGB = 25
if ($TextOnly) { $needGB = 24 }
if ($SkipModel) { $needGB = 5 }
if ($freeGB -lt $needGB) {
    Write-Host "  [WARN] Free disk space is only $([math]::Round($freeGB,1)) GB; at least $needGB GB recommended. / 警告：磁盘剩余 $([math]::Round($freeGB,1)) GB，建议至少 $needGB GB。" -ForegroundColor Yellow
}

if ($Proxy) {
    Write-Info "Using proxy: $Proxy" "使用代理：$Proxy"
}
if ($TextOnly) {
    Write-Info "Text-only mode: vision projector (mmproj) will NOT be downloaded." "纯文本模式：不下载视觉编码器（mmproj）。"
}

# ---------- Step 2: download llama.cpp ----------
Write-Step "Downloading llama.cpp (b11491, CUDA 13.4)..." "下载 llama.cpp 引擎（b11491, CUDA 13.4）..."
New-Item -ItemType Directory -Path $llamaCppDir -Force | Out-Null
if ((Test-Path $llamaServerExe) -and -not $Force) {
    Write-Info "llama-server.exe already exists, skipping. (use -Force to re-download)" "已存在 llama-server.exe，跳过（-Force 强制重下）"
} else {
    $zip = Join-Path $env:TEMP "llama-b11491-bin-win-cuda-13.4-x64.zip"
    $proxyArg = @(); if ($Proxy) { $proxyArg = @("-x", $Proxy) }
    & curl.exe @proxyArg -L -C - --http1.1 --retry 5 --retry-all-errors --fail -o $zip $LlamaCppUrl
    if ($LASTEXITCODE -ne 0) { Write-Error "llama.cpp download failed (exit $LASTEXITCODE). / 引擎下载失败。" }
    Write-Info "Extracting..." "解压中..."
    Expand-Archive -Path $zip -DestinationPath $llamaCppDir -Force
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path $llamaServerExe)) { Write-Error "llama-server.exe missing after extraction. / 解压后未找到 llama-server.exe。" }
}

# ---------- Step 3: download model (+ mmproj unless text-only) ----------
Write-Step "Downloading Qwen3.8-27B IQ3_S-mtp GGUF (resumable)..." "下载模型权重 IQ3_S-mtp（断点续传）..."
New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
if ($SkipModel) {
    Write-Info "Skipped (as requested). / 已按参数跳过模型下载。"
} else {
    if ((Test-Path $modelFile) -and -not $Force) {
        Write-Info "Model already exists, skipping. (use -Force to re-download)" "已存在模型文件，跳过（-Force 强制重下）"
    } else {
        $bytes = Invoke-ResumableDownload -Url $ModelUrl -OutFile $modelFile -ExpectedSize $modelSize
        $sizeGB = [math]::Round($bytes / 1e9, 2)
        Write-Info "Model downloaded: $sizeGB GB ($bytes bytes)" "模型下载完成：$sizeGB GB（$bytes 字节）"
        if ($bytes -lt 11000000000) {
            Write-Host "  [WARN] Model file is only $sizeGB GB. The download may be incomplete; re-run to resume. / 警告：模型文件仅 $sizeGB GB，下载可能不完整；重跑可续传。" -ForegroundColor Yellow
        }
    }
    if ($TextOnly) {
        Write-Info "Vision projector skipped (-TextOnly)." "已跳过视觉编码器（-TextOnly）。"
    } elseif ((Test-Path $mmprojFile) -and -not $Force) {
        Write-Info "mmproj already exists, skipping. (use -Force to re-download)" "已存在 mmproj，跳过（-Force 强制重下）"
    } else {
        $bytes = Invoke-ResumableDownload -Url $MmprojUrl -OutFile $mmprojFile -ExpectedSize $mmprojSize
        Write-Info "mmproj downloaded: $bytes bytes" "视觉编码器下载完成：$bytes 字节"
        if ($bytes -lt 900000000) {
            Write-Host "  [WARN] mmproj file is only $bytes bytes; may be incomplete. / 警告：mmproj 仅 $bytes 字节，可能不完整。" -ForegroundColor Yellow
        }
    }
}

# ---------- Step 4: verify ----------
Write-Step "Verifying installation..." "验证安装..."
$ok = $true
if (-not (Test-Path $llamaServerExe)) { Write-Host "  [FAIL] llama-server.exe missing / 缺少 llama-server.exe" -ForegroundColor Red; $ok = $false }
else { Write-Info "llama.cpp: OK" "引擎：OK" }
if (-not $SkipModel) {
    if (-not (Test-Path $modelFile)) { Write-Host "  [FAIL] model missing / 缺少模型文件" -ForegroundColor Red; $ok = $false }
    else {
        $modelSizeGB = [math]::Round((Get-Item $modelFile).Length / 1e9, 2)
        Write-Info "model: $modelSizeGB GB" "模型：$modelSizeGB GB"
        if ((Get-Item $modelFile).Length -lt 11000000000) { Write-Host "  [WARN] model size $modelSizeGB GB looks incomplete / 警告：模型大小 $modelSizeGB GB 可能不完整" -ForegroundColor Yellow }
    }
    if (-not $TextOnly) {
        if (-not (Test-Path $mmprojFile)) { Write-Host "  [FAIL] mmproj missing / 缺少视觉编码器" -ForegroundColor Red; $ok = $false }
        else { Write-Info "mmproj: $((Get-Item $mmprojFile).Length) bytes" "视觉编码器：$((Get-Item $mmprojFile).Length) 字节" }
    }
}
if (-not $ok) { Write-Error "Installation incomplete. / 安装不完整。" }

# ---------- Step 5: done ----------
Write-Step "Done! Next steps:" "完成！下一步："
$startScript = if ($TextOnly) { 'start-server-iq3s-q8kv-text-only-256k.ps1' } else { 'start-server-iq3s-q4kv-text-image-256k.ps1' }
Write-Host ""
Write-Host "  Start the server / 启动服务:" -ForegroundColor Green
Write-Host "    powershell -ExecutionPolicy Bypass -File $(Join-Path $scriptDir $startScript)"
Write-Host ""
Write-Host "  Verify / 验证:" -ForegroundColor Green
Write-Host "    powershell -ExecutionPolicy Bypass -File $(Join-Path $scriptDir 'verify-server.ps1')"
Write-Host ""
Write-Host "  Benchmark / 基准测试:" -ForegroundColor Green
Write-Host "    powershell -NoProfile -File $(Join-Path $scriptDir 'benchmark.ps1')"
Write-Host ""
Write-Host "  Server endpoint / 服务地址: http://127.0.0.1:8080/v1 (OpenAI-compatible)"
