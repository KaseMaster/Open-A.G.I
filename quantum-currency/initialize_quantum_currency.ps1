# initialize_quantum_currency.ps1
# GHSP v1.2 Quantum Currency Initialization for Windows

Write-Host "=== GHSP v1.2 Quantum Currency Initialization ==="
Write-Host "[INIT] Launching Quantum Currency Integration Directive (QCI–HSMF v1.2)"

# Activate HARU dynamic feedback learning
Write-Host "[STEP 1] Activating HARU dynamic feedback learning"
python haru/autoregression.py --init --cycles 150 --verify_lambda_convergence
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] HARU initialization failed"
    exit 1
}

# Establish field-level coherence engine
Write-Host "[STEP 2] Establishing field-level coherence engine"
python src/core/coherence_engine.py --mode initialize --phi-damping
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Coherence engine initialization failed"
    exit 1
}

# Initialize the gating service with adaptive thresholds
Write-Host "[STEP 3] Initializing gating service with adaptive thresholds"
python src/core/gating_service.py --adaptive --verify-safemode
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Gating service initialization failed"
    exit 1
}

# Sync recursive coherence memory and heatmap dashboard
Write-Host "[STEP 4] Syncing recursive coherence memory"
python src/core/memory.py --sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Memory sync failed"
    exit 1
}

Write-Host "[STEP 5] Streaming curvature dashboard"
python src/api/routes/curvature.py --stream --verify
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Curvature dashboard streaming failed"
    exit 1
}

# Start full continuous attunement verification cycle
Write-Host "[STEP 6] Starting full continuous attunement verification cycle"
powershell -File ci/verify_quantum_currency_coherence.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Verification pipeline failed"
    exit 1
}

Write-Host "✅ GHSP v1.2 Continuous Attunement Active"