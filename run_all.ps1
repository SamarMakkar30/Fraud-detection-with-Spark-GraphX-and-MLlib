# run_all.ps1 — Full Pipeline Orchestrator
# INT315 Cluster Computing
# Runs all 4 stages sequentially with error checking

$pythonExe = (Get-Command python -ErrorAction Stop).Source
$env:PYSPARK_PYTHON = $pythonExe
$env:PYSPARK_DRIVER_PYTHON = $pythonExe
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$ErrorActionPreference = "Continue"

# Ensure we are in the project directory
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if ($projectDir -and (Test-Path $projectDir)) {
    Set-Location $projectDir
}

Write-Host "  Working directory: $(Get-Location)"

$startTime = Get-Date
$stageResults = @()
$pipelineFailed = $false

function Run-Stage {
    param(
        [string]$StageName,
        [scriptblock]$ScriptBlock
    )
    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host "  RUNNING: $StageName"
    Write-Host ("=" * 60)
    Write-Host ""

    $stageStart = Get-Date

    # Execute the command; $ErrorActionPreference = Continue prevents
    # PowerShell from treating stderr (Java warnings) as terminating.
    & $ScriptBlock
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) { $exitCode = 0 }

    $stageEnd = Get-Date
    $elapsed = ($stageEnd - $stageStart).TotalSeconds

    $status = if ($exitCode -eq 0) { "PASS" } else { "FAIL" }

    Write-Host ""
    Write-Host "  [$status] $StageName completed in $([math]::Round($elapsed, 1))s (exit code: $exitCode)"

    $script:stageResults += [PSCustomObject]@{
        Stage    = $StageName
        Status   = $status
        ExitCode = $exitCode
        TimeSec  = [math]::Round($elapsed, 1)
    }

    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host "  PIPELINE ABORTED -- $StageName failed with exit code $exitCode!"
        $script:pipelineFailed = $true
        return $false
    }
    return $true
}

# ── Stage 1: Data Generation ────────────────────────────────────────
$ok = Run-Stage "Stage 1: Data Generation (Python)" {
    python src/generate_data.py
}
if (-not $ok) { exit 1 }

# ── Stage 2: Spark ETL (RDD + SQL) ──────────────────────────────────
$ok = Run-Stage "Stage 2: Spark ETL (PySpark)" {
    spark-submit src/spark_etl.py
}
if (-not $ok) {
    throw "Stage 2 failed; stopping pipeline."
}

# ── Stage 3: GraphX Analytics (Scala) ───────────────────────────────
$ok = Run-Stage "Stage 3: GraphX Analytics (Scala)" {
    spark-shell -i src/GraphAnalytics.scala
}
if (-not $ok) {
    throw "Stage 3 failed; stopping pipeline."
}

# ── Stage 4: ML Pipeline ────────────────────────────────────────────
$ok = Run-Stage "Stage 4: ML Pipeline (PySpark MLlib)" {
    spark-submit src/ml_pipeline.py
}
if (-not $ok) {
    throw "Stage 4 failed; stopping pipeline."
}

# ── Final Summary ───────────────────────────────────────────────────
$endTime = Get-Date
$totalElapsed = ($endTime - $startTime).TotalSeconds

Write-Host ""
Write-Host ("=" * 60)
Write-Host "  PIPELINE EXECUTION SUMMARY"
Write-Host ("=" * 60)
Write-Host ""

$stageResults | Format-Table -AutoSize

Write-Host "  Total pipeline time: $([math]::Round($totalElapsed, 1))s"
Write-Host ""

# Verify all outputs
Write-Host "  OUTPUT FILES:"
$outputFiles = @(
    "data/raw/accounts.csv",
    "data/raw/transactions.csv",
    "data/processed/sql_features.parquet/_SUCCESS",
    "data/processed/graph_features.parquet/_SUCCESS",
    "outputs/confusion_matrix.png",
    "outputs/correlation_heatmap.png",
    "outputs/ablation_comparison.png",
    "outputs/benchmark_results.csv"
)

$allExist = $true
foreach ($f in $outputFiles) {
    if (Test-Path $f) {
        Write-Host "    OK  $f"
    } else {
        Write-Host "    MISSING  $f"
        $allExist = $false
    }
}

Write-Host ""
if ($allExist -and -not $pipelineFailed) {
    Write-Host "  ALL STAGES PASSED -- Pipeline completed successfully!"
} else {
    Write-Host "  WARNING: Some output files are missing or stages failed."
}
Write-Host ("=" * 60)
