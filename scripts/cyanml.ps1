param(
    [ValidateSet("start", "stop", "restart", "status")]
    [string]$Command = "start"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$stateDirectory = Join-Path $projectRoot ".cyanml"
$backendPidFile = Join-Path $stateDirectory "backend.pid"
$frontendPidFile = Join-Path $stateDirectory "frontend.pid"

function Get-ManagedProcess([string]$pidFile) {
    if (-not (Test-Path -LiteralPath $pidFile)) {
        return $null
    }
    $managedProcessId = [int](Get-Content -LiteralPath $pidFile -Raw)
    return Get-Process -Id $managedProcessId -ErrorAction SilentlyContinue
}

function Stop-ManagedProcess([string]$name, [string]$pidFile) {
    $managedProcess = Get-ManagedProcess $pidFile
    if ($null -ne $managedProcess) {
        Stop-Process -Id $managedProcess.Id -Force
        $managedProcess.WaitForExit(5000) | Out-Null
        Write-Host "Stopped $name (PID $($managedProcess.Id))."
    }
    if (Test-Path -LiteralPath $pidFile) {
        Remove-Item -LiteralPath $pidFile -Force
    }
}

function Stop-CyanML {
    Stop-ManagedProcess "frontend" $frontendPidFile
    Stop-ManagedProcess "backend" $backendPidFile
}

function Start-CyanML {
    New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null
    if ((Get-ManagedProcess $backendPidFile) -or (Get-ManagedProcess $frontendPidFile)) {
        throw "CyanML is already running. Use 'cyanml.bat status' or 'cyanml.bat stop'."
    }

    $pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        Push-Location $projectRoot
        try {
            & uv sync --python 3.12
            if ($LASTEXITCODE -ne 0) { throw "uv sync failed." }
        }
        finally { Pop-Location }
    }

    $nodeModules = Join-Path $projectRoot "frontend\node_modules"
    if (-not (Test-Path -LiteralPath $nodeModules)) {
        Push-Location (Join-Path $projectRoot "frontend")
        try {
            & npm.cmd install
            if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
        }
        finally { Pop-Location }
    }

    $backendProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList "-m", "uvicorn", "backend.cyanml_api.app:app", "--host", "127.0.0.1", "--port", "8765" `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput (Join-Path $stateDirectory "backend.log") `
        -RedirectStandardError (Join-Path $stateDirectory "backend.error.log") `
        -WindowStyle Hidden `
        -PassThru

    $viteScript = Join-Path $projectRoot "frontend\node_modules\vite\bin\vite.js"
    $frontendProcess = Start-Process `
        -FilePath "node.exe" `
        -ArgumentList $viteScript, "--host", "127.0.0.1", "--port", "5173", "--strictPort" `
        -WorkingDirectory (Join-Path $projectRoot "frontend") `
        -RedirectStandardOutput (Join-Path $stateDirectory "frontend.log") `
        -RedirectStandardError (Join-Path $stateDirectory "frontend.error.log") `
        -WindowStyle Hidden `
        -PassThru

    Set-Content -LiteralPath $backendPidFile -Value $backendProcess.Id -Encoding ascii
    Set-Content -LiteralPath $frontendPidFile -Value $frontendProcess.Id -Encoding ascii
    Write-Host "CyanML started."
    Write-Host "Frontend: http://127.0.0.1:5173"
    Write-Host "Backend:  http://127.0.0.1:8765"
}

function Show-Status {
    $backendProcess = Get-ManagedProcess $backendPidFile
    $frontendProcess = Get-ManagedProcess $frontendPidFile
    Write-Host "Backend:  $(if ($backendProcess) { "running (PID $($backendProcess.Id))" } else { "stopped" })"
    Write-Host "Frontend: $(if ($frontendProcess) { "running (PID $($frontendProcess.Id))" } else { "stopped" })"
}

switch ($Command) {
    "start" { Start-CyanML }
    "stop" { Stop-CyanML }
    "restart" { Stop-CyanML; Start-CyanML }
    "status" { Show-Status }
}
