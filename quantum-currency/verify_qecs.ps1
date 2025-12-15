# verify_qecs.ps1
# QECS Full Coherence & Transaction Verification

Write-Host "[VERIFY] Running QECS Full Coherence & Transaction Verification"

# Integration Tests
Write-Host "[STEP 1] Running Integration Tests"
python ci/test_quantum_currency_integration.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Integration tests failed"
    exit 1
}

# Verify Coherence & QRA
Write-Host "[STEP 2] Verifying Coherence & QRA"
bash ci/verify_quantum_currency_coherence.sh
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Coherence verification failed"
    exit 1
}

# Windows support
Write-Host "[STEP 3] Running Windows verification"
powershell -File ci/verify_quantum_currency_coherence.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Windows verification failed"
    exit 1
}

Write-Host "[VERIFY] QECS v1.3 Fully Operational"
Write-Host ""
Write-Host "Expected Outcomes:"
Write-Host "  System coherence maximized (C_system → 1)"
Write-Host "  Action efficiency minimized (I_eff → 0)"
Write-Host "  GAS_target ≥ 0.99"
Write-Host "  λ_opt(L) self-tuned"
Write-Host "  QRA keys fully integrated for identity and gating"
Write-Host "  CAF emission policy active"
Write-Host "  Field-level security with Gravity Well detection active"
Write-Host "  Dashboard real-time verification visualized"