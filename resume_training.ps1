<#
.SYNOPSIS
    Автоматичний запуск тренування з resume з останнього чекпойнта.
    Читає train_config.json, знаходить останній model/gpt_model.bin + gpt_opt.bin,
    запускає shade.exe з правильними параметрами, логує, комітить в гіт.

.DESCRIPTION
    Плагін для роботи на слабкому ноуті:
    - Підлаштовує ctx/batch/grad_accum під "laptop" секцію конфігу
    - Авто-знаходить останній чекпойнт
    - Запускає з --continue
    - Записує лог в build/train_<дата>_<час>.log
    - Після завершення комітить лог + зміни в гіт з повідомленням про епокхи

.USAGE
    .\resume_training.ps1           # З дефолтними налаштуваннями з train_config.json
    .\resume_training.ps1 -Mode Chat   # SFT чат
    .\resume_training.ps1 -Mode DPO    # DPO
    .\resume_training.ps1 -Epochs 10   # Перевизначити епокхи
#>

param(
    [ValidateSet('GPT','Chat','DPO')]
    [string]$Mode = 'GPT',
    [int]$Epochs = 0,           # 0 = з конфігу
    [int]$Batch = 0,            # 0 = з конфігу
    [int]$Ctx = 0,              # 0 = з конфігу
    [switch]$NoDashboard,
    [switch]$NoGitCommit
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot)

# --- Load config ---
$configPath = "train_config.json"
if (-not (Test-Path $configPath)) { Write-Error "train_config.json not found"; exit 1 }
$cfg = Get-Content $configPath -Raw | ConvertFrom-Json

# --- Select mode config ---
switch ($Mode) {
    'GPT'  { $mc = $cfg.train_gpt;  $cmd = 'train gpt';  $dataArg = '--data', $cfg.data.train_file }
    'Chat' { $mc = $cfg.train_chat; $cmd = 'train chat'; $dataArg = '--data', $cfg.data.dialogues_file }
    'DPO'  { $mc = $cfg.train_dpo;  $cmd = 'train dpo';  $dataArg = '--data', $cfg.data.prefs_file }
    default { Write-Error "Unknown mode $Mode"; exit 1 }
}

# --- Override with laptop settings if on laptop (detect by hostname or env) ---
$isLaptop = $env:COMPUTERNAME -match 'LAPTOP|NOTEBOOK|MOBILE' -or $env:SHADE_LAPTOP
if ($isLaptop -and $cfg.laptop) {
    Write-Host "🖥 Laptop mode detected — applying laptop overrides" -ForegroundColor Cyan
    if ($cfg.laptop.ctx)     { $mc.ctx = $cfg.laptop.ctx }
    if ($cfg.laptop.batch)   { $mc.batch = $cfg.laptop.batch }
    if ($cfg.laptop.grad_accum) { $mc.grad_accum = $cfg.laptop.grad_accum }
    if ($cfg.laptop.lr_scale)  { $mc.lr = [double]$mc.lr * $cfg.laptop.lr_scale }
}

# --- Override from CLI args ---
if ($Epochs -gt 0) { $mc.epochs = $Epochs }
if ($Batch -gt 0)  { $mc.batch = $Batch }
if ($Ctx -gt 0)    { $mc.ctx = $Ctx }

# --- Build shade.exe path ---
$shadeExe = "build\shade.exe"
if (-not (Test-Path $shadeExe)) { Write-Error "shade.exe not found at $shadeExe"; exit 1 }

# --- Check checkpoint ---
$modelDir = $cfg.checkpoints.model_dir
$resumeModel = Join-Path $modelDir $cfg.checkpoints.resume_model
$optState = Join-Path $modelDir $cfg.checkpoints.opt_state
$hasResume = (Test-Path $resumeModel) -and (Test-Path $optState)

# --- Build arguments ---
$args = @($cmd)
$args += $dataArg
$args += '--epochs', $mc.epochs
$args += '--ctx', $mc.ctx
$args += '--dim', $cfg.model.dim
$args += '--blocks', $cfg.model.blocks
$args += '--lr', $mc.lr
if ($mc.batch) { $args += '--batch', $mc.batch }
if ($mc.grad_accum) { $args += '--grad-accum', $mc.grad_accum }  # буде додано в код
if ($hasResume) { $args += '--continue' }
if (-not $NoDashboard -and $cfg.dashboard.enabled) { $args += '--dashboard', '--port', $cfg.dashboard.port }
if ($Mode -eq 'DPO' -and $cfg.train_dpo.beta) { $args += '--beta', $mc.beta }
if ($Mode -eq 'DPO' -and $cfg.train_dpo.ref_model) { $args += '--ref-model', $cfg.train_dpo.ref_model }

Write-Host "=== ShadeAI Resume Training ===" -ForegroundColor Green
Write-Host "Mode: $Mode"
Write-Host "Epochs: $($mc.epochs), Ctx: $($mc.ctx), Batch: $($mc.batch), LR: $($mc.lr)"
Write-Host "Resume: $hasResume (model: $resumeModel)"
Write-Host "Args: $($args -join ' ')"
Write-Host ""

# --- Log file ---
$logDir = $cfg.logging.log_dir
$prefix = $cfg.logging.log_prefix
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "$prefix`_$timestamp.log"
$errFile  = Join-Path $logDir "$prefix`_$timestamp.err"

# --- Run ---
Write-Host "Starting: $shadeExe $($args -join ' ')" -ForegroundColor Yellow
Write-Host "Log: $logFile"
Write-Host ""

$pinfo = Start-Process -FilePath $shadeExe -ArgumentList $args -WorkingDirectory "build" `
    -RedirectStandardOutput $logFile -RedirectStandardError $errFile -PassThru -WindowStyle Hidden

Write-Host "PID: $($pinfo.Id)" -ForegroundColor Cyan
Write-Host "Log: $logFile"
Write-Host "Waiting for completion..." -ForegroundColor Gray

$pinfo.WaitForExit()
$exitCode = $pinfo.ExitCode

# --- Read last lines of log ---
$lastLines = Get-Content $logFile -Tail 20 -ErrorAction SilentlyContinue
Write-Host "`n=== Last log lines ===" -ForegroundColor Gray
$lastLines | ForEach-Object { Write-Host $_ }

# --- Git commit ---
if (-not $NoGitCommit) {
    Write-Host "`nCommitting to git..." -ForegroundColor Cyan
    git add -A
    $msg = "train($Mode): ep$($mc.epochs) ctx$($mc.ctx) lr$($mc.lr) exit=$exitCode`n$(($lastLines[-5..-1] -join '`n'))"
    git commit -m $msg -q
    Write-Host "Committed." -ForegroundColor Green
}

exit $exitCode