#Requires -Version 7
<#
.SYNOPSIS
    Long-context (196K / 262K) benchmark for Qwen3.8-27B on llama.cpp.
    196K / 262K 长上下文基准测试。

.DESCRIPTION
    Verifies stability and measures TTFT / prefill / decode / VRAM at very long contexts
    (beyond benchmark.ps1's 128K) on high-VRAM GPUs such as the RTX PRO 4500 (32 GB).
    验证高显存显卡（如 RTX PRO 4500 32GB）上 128K 以上长上下文的稳定性，并测量 TTFT/prefill/解码/显存。

.PARAMETER Contexts
    Comma-separated list of context token targets. 上下文 token 目标列表。

.PARAMETER GenTokens
    Number of tokens to generate per scenario. 每场景生成 token 数。

.EXAMPLE
    .\benchmark-full.ps1
    .\benchmark-full.ps1 -Contexts 196000,262000 -GenTokens 64
#>
param(
    [int[]]$Contexts = @(196000, 262000),
    [int]$GenTokens = 64
)

$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8080/v1"
$model = "qwen3.8-27b"

function Get-TokenEstimate([string]$text) {
    return [int]($text.Length / 5.5)
}

$unit = @"
The llama.cpp project provides a minimal C++ API for running large language models locally. It supports many quantization formats and can offload computation to GPUs through CUDA. Tensor parallelism splits the model across multiple graphics cards, trading some communication overhead for higher memory capacity and faster compute. KV cache quantization reduces memory pressure at the cost of small accuracy degradation, and modern builds expose these knobs through command-line flags. Server mode speaks the OpenAI-compatible protocol, so clients such as opencode can point their base URL at a local endpoint. When benchmarking, keep the prompt cache in mind, because repeated prefixes are evaluated once and then reused on later requests. Generation speed is dominated by memory bandwidth, while prompt evaluation is dominated by compute throughput. A single RTX A5000 delivers roughly nine hundred gigabytes per second of bandwidth, which yields about thirty tokens per second for a twenty-seven billion parameter model quantized to Q4. Adding a second card approximately doubles the aggregate bandwidth, but PCIe synchronization and smaller per-GPU layer slices usually cut the theoretical gain by twenty to forty percent. Context length is bounded by the product of cache size and quantization level; larger contexts need smaller caches or more video memory. The best measurement practice is to send several uncached prompts of varying length and report the median, since the first run can be warmer than later ones. This paragraph is designed to be repeated so that it produces a stable and predictable token count for benchmarking purposes.
"@

function New-Prompt([int]$target, [string]$seed) {
    $text = "Message-Id: $seed`n`n"
    while ((Get-TokenEstimate $text) -lt $target) {
        $text += $unit + "`n"
    }
    return $text
}

function Invoke-Run([string]$prompt, [int]$maxTokens) {
    $payload = @{
        model           = $model
        messages        = @(@{ role = "user"; content = $prompt })
        max_tokens      = $maxTokens
        stream          = $true
        stream_options  = @{ include_usage = $true }
    } | ConvertTo-Json -Depth 5 -Compress

    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(20)
    $req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$base/chat/completions")
    $req.Content = [System.Net.Http.StringContent]::new($payload, [System.Text.Encoding]::UTF8, "application/json")

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = $null; $stream = $null; $reader = $null
    $timings = $null; $usage = $null; $ttft = $null; $first = $false
    try {
        $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        $stream = $resp.Content.ReadAsStreamAsync().Result
        $reader = [System.IO.StreamReader]::new($stream)
        while (($line = $reader.ReadLine()) -ne $null) {
            if (-not $line.StartsWith("data: ")) { continue }
            $json = $line.Substring(6)
            if ($json -eq "[DONE]") { break }
            $obj = $json | ConvertFrom-Json
            if (-not $first) {
                $first = $true
                $sw.Stop(); $ttft = $sw.Elapsed.TotalSeconds; $sw.Start()
            }
            if ($obj.timings) { $timings = $obj.timings }
            if ($obj.usage) { $usage = $obj.usage }
        }
    } finally {
        $sw.Stop()
        if ($reader) { $reader.Dispose() }
        if ($stream) { $stream.Dispose() }
        if ($resp) { $resp.Dispose() }
        $client.Dispose()
    }
    return [pscustomobject]@{
        ttft    = $ttft
        total   = $sw.Elapsed.TotalSeconds
        usage   = $usage
        timings = $timings
    }
}

function Show-Run([pscustomobject]$r, [string]$label) {
    if (-not $r.timings) { Write-Host "$label : NO TIMINGS"; return }
    $t = $r.timings
    Write-Host "$label"
    Write-Host ("  prompt eval : n={0} (cache {1})  {2:N1} ms  ->  {3:N1} tok/s" -f $t.prompt_n, $t.cache_n, $t.prompt_ms, $t.prompt_per_second)
    Write-Host ("  generation  : n={0}               {1:N1} ms  ->  {2:N1} tok/s  ({3:N1} ms/tok)" -f $t.predicted_n, $t.predicted_ms, $t.predicted_per_second, $t.predicted_per_token_ms)
    Write-Host ("  TTFT        : {0:N2} s    total: {1:N2} s" -f $r.ttft, $r.total)
}

Write-Host "== Environment =="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
Get-Process llama-server -ErrorAction SilentlyContinue | ForEach-Object {
    $uptime = try { ((Get-Date) - $_.StartTime).TotalSeconds } catch { 0 }
    [pscustomobject]@{ Id = $_.Id; RAM_GB = [math]::Round($_.WorkingSet64 / 1GB, 2); CPU_pct = [math]::Round($_.CPU / [Math]::Max($uptime, 1) * 100, 0) }
} | Format-Table -AutoSize

$results = @()
foreach ($target in $Contexts) {
    $label = "{0}K" -f [math]::Round($target / 1000)
    Write-Host "`n== Scenario: prompt eval ~$label tokens (uncached) =="
    $r = Invoke-Run (New-Prompt $target ([guid]::NewGuid().ToString())) $GenTokens
    Show-Run $r "  ${label} prompt + $GenTokens gen"
    Write-Host "  --- GPU after scenario ---"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
    $results += [pscustomobject]@{
        Target  = $target
        PromptN = $r.timings.prompt_n
        Prefill = $r.timings.prompt_per_second
        Gen     = $r.timings.predicted_per_second
        Draft   = "{0}/{1}" -f $r.timings.draft_n_accepted, $r.timings.draft_n
        TTFT    = $r.ttft
    }
}

Write-Host "`n== Summary =="
$results | Format-Table -AutoSize @{n="ctx";           e={$_.Target}},
                                      @{n="prompt_n";     e={$_.PromptN}},
                                      @{n="prefill tok/s"; e={"{0:N1}" -f $_.Prefill}},
                                      @{n="gen tok/s";    e={"{0:N1}" -f $_.Gen}},
                                      @{n="draft acc";    e={$_.Draft}},
                                      @{n="TTFT s";       e={"{0:N2}" -f $_.TTFT}}

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader