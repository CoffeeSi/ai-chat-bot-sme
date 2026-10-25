param([switch]$PullModel)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
# Keep .env parsing consistent with the Python application, including BOM and quotes.
$taskConfig = & uv run python -c 'import json,os; from rmt_assignment3_project.config import rag; print(json.dumps(dict(provider=os.getenv(''RAG_LLM_PROVIDER''),model=os.getenv(''RAG_LLM_MODEL'',''kazllm:latest''),url=os.getenv(''OLLAMA_BASE_URL'',''http://127.0.0.1:11434''))))'
if ($LASTEXITCODE -ne 0) { throw 'Cannot read local model configuration. Run uv sync first.' }
$taskSettings = $taskConfig | ConvertFrom-Json
if ($taskSettings.provider -ne 'ollama') { throw 'Set RAG_LLM_PROVIDER=ollama in .env first.' }
$taskOllama = Get-Command ollama -ErrorAction SilentlyContinue
$taskOllamaPath = if ($taskOllama) { $taskOllama.Source } else { Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
$taskUrl = $taskSettings.url.TrimEnd('/')
try {
    Invoke-RestMethod -Uri "$taskUrl/api/tags" -TimeoutSec 5 | Out-Null
} catch {
    if ($taskUrl -notin @('http://127.0.0.1:11434','http://localhost:11434')) {
        throw "Ollama is unavailable at $taskUrl. Start that server first."
    }
    if (-not (Test-Path -LiteralPath $taskOllamaPath)) {
        throw 'Install Ollama from https://ollama.com/download/windows and rerun this script.'
    }
    Start-Process -FilePath $taskOllamaPath -ArgumentList 'serve' -WindowStyle Hidden
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 30; $taskAttempt++) {
        Start-Sleep -Seconds 1
        try {
            Invoke-RestMethod -Uri "$taskUrl/api/tags" -TimeoutSec 2 | Out-Null
            $taskReady = $true
            break
        } catch {}
    }
    if (-not $taskReady) { throw 'Ollama did not start. Open the Ollama application and retry.' }
}
if ($PullModel) {
    # Use the configured server, including when it differs from the local default.
    if (-not (Test-Path -LiteralPath $taskOllamaPath)) { throw 'Ollama CLI is needed to download the model.' }
    $taskPreviousHost = $env:OLLAMA_HOST
    try {
        $env:OLLAMA_HOST = $taskUrl
        & $taskOllamaPath pull $taskSettings.model
        if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
    } finally {
        $env:OLLAMA_HOST = $taskPreviousHost
    }
}
try {
    $taskBody = @{ model = $taskSettings.model } | ConvertTo-Json
    Invoke-RestMethod -Method Post -Uri "$taskUrl/api/show" -ContentType 'application/json' -Body $taskBody -TimeoutSec 10 | Out-Null
} catch {
    throw 'Model is unavailable. Import your KazLLM GGUF as kazllm using ollama create kazllm -f Modelfile. For registry models use -PullModel.'
}
Write-Host "Local model: $($taskSettings.model)"
Write-Host 'Chat: http://127.0.0.1:8000 (wait for Application startup complete)'
& uv run uvicorn rmt_assignment3_project.api.api:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
