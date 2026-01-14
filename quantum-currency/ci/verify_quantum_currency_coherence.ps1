# ci/verify_quantum_currency_coherence.ps1
# Quantum Currency Integration Verification Pipeline for Windows

Write-Host "[VERIFY] Starting Quantum Currency Coherence Pipeline"

python ci/simulate_unattunement.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] simulate_unattunement.py failed"
    exit 1
}

python ci/verify_gas_threshold.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] verify_gas_threshold.py failed"
    exit 1
}

python ci/verify_metrics.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] verify_metrics.py failed"
    exit 1
}

python ci/test_safe_mode.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] test_safe_mode.py failed"
    exit 1
}

python ci/validate_heatmap.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] validate_heatmap.py failed"
    exit 1
}

Write-Host "✅ All Quantum Currency Coherence Tests Passed"