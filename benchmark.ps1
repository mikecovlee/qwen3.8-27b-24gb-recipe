#Requires -Version 7
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
    $client.Timeout = [TimeSpan]::FromMinutes(10)
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

Write-Host "`n== Scenario 1: prompt eval ~2K tokens (uncached) =="
$r1 = Invoke-Run (New-Prompt 2000 ([guid]::NewGuid().ToString())) 16
Show-Run $r1 "  2K prompt + 16 gen"

Write-Host "`n== Scenario 2: prompt eval ~4K tokens (uncached) =="
$r2 = Invoke-Run (New-Prompt 4000 ([guid]::NewGuid().ToString())) 16
Show-Run $r2 "  4K prompt + 16 gen"

Write-Host "`n== Scenario 3: generation 512 tokens (x2 runs) =="
$r3a = Invoke-Run ("Message-Id: $([guid]::NewGuid().ToString())`n`nWrite a long, detailed technical article about running local large language models on consumer GPUs. Cover quantization, KV cache, context length, benchmarking methodology, and practical tuning tips. Be thorough and write at least eight hundred words.") 512
Show-Run $r3a "  run 1"
$r3b = Invoke-Run ("Message-Id: $([guid]::NewGuid().ToString())`n`nWrite a long, detailed technical article about running local large language models on consumer GPUs. Cover quantization, KV cache, context length, benchmarking methodology, and practical tuning tips. Be thorough and write at least eight hundred words.") 512
Show-Run $r3b "  run 2"

Write-Host "`n== Scenario 4: generation 256 tokens at ~64K context =="
$r4 = Invoke-Run (New-Prompt 64000 ([guid]::NewGuid().ToString())) 256
Show-Run $r4 "  64K prompt + 256 gen"

Write-Host "`n== Scenario 5: generation 128 tokens at ~128K context =="
$r5 = Invoke-Run (New-Prompt 128000 ([guid]::NewGuid().ToString())) 128
Show-Run $r5 "  128K prompt + 128 gen"

Write-Host "`n== Summary =="
Write-Host ("  prompt eval: 2K -> {0:N1} tok/s   4K -> {1:N1} tok/s" -f $r1.timings.prompt_per_second, $r2.timings.prompt_per_second)
Write-Host ("  generation : run1 {0:N1} tok/s   run2 {1:N1} tok/s" -f $r3a.timings.predicted_per_second, $r3b.timings.predicted_per_second)
Write-Host ("  generation @64K ctx: {0:N1} tok/s (draft {1:N0}/{2:N0})" -f $r4.timings.predicted_per_second, $r4.timings.draft_n_accepted, $r4.timings.draft_n)
Write-Host ("  generation @128K ctx: {0:N1} tok/s (draft {1:N0}/{2:N0})" -f $r5.timings.predicted_per_second, $r5.timings.draft_n_accepted, $r5.timings.draft_n)
Write-Host ("  TTFT       : run1 {0:N2} s   run2 {1:N2} s   64K: {2:N2} s   128K: {3:N2} s" -f $r3a.ttft, $r3b.ttft, $r4.ttft, $r5.ttft)
Write-Host ("  prompt_n   : run1 {0}   64K {1}   128K {2}   (actual token counts)" -f $r3a.timings.prompt_n, $r4.timings.prompt_n, $r5.timings.prompt_n)
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
