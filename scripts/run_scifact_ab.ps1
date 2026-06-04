# SciFact quality A/B: Signal 5b off (λ=0) vs on (λ=0.35)
# Requires BEIR scifact under beir_data_root (see beir_data_root.py).
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path logs | Out-Null

function Run-Arm {
    param([string]$Name, [double]$Lambda)
    $log = Join-Path $root "logs\scifact_quality_lambda_$Name.log"
    Write-Host "=== $Name (lambda-pf=$Lambda) -> $log"
    & python -u eval_beir.py --datasets scifact --mode quality --lambda-pf $Lambda 2>&1 |
        Tee-Object -FilePath $log
    Select-String -Path $log -Pattern "NDCG@10|R@10|BM25 ref"
}

Write-Host "Baseline (Signal 5b off)"
Run-Arm -Name "0" -Lambda 0.0
Write-Host "Treatment (Signal 5b on)"
Run-Arm -Name "035" -Lambda 0.35
Write-Host "Done. Compare NDCG@10 lines in logs/"
