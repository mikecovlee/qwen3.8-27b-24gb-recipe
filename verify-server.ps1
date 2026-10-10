$ErrorActionPreference = "Stop"

Write-Host "=== Health Check ===" -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -Method Get -TimeoutSec 10
    Write-Host "Status: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "Server not responding. Is llama-server running?" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Chat Test ===" -ForegroundColor Cyan
$body = @{model="qwen3.8-27b"; messages=@(@{role="user"; content="用一句话介绍你自己"}); max_tokens=128} | ConvertTo-Json -Depth 5
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 120
Write-Host "Model: $($response.model)" -ForegroundColor Gray
Write-Host "Response: $($response.choices[0].message.content)" -ForegroundColor Green
Write-Host "Tokens: prompt=$($response.usage.prompt_tokens) completion=$($response.usage.completion_tokens) total=$($response.usage.total_tokens)" -ForegroundColor Gray

Write-Host "`n=== GPU Usage ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
