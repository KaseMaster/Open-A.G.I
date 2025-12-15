# initialize_qecs.ps1
# QECS v1.3 Quantum Currency Initialization

Write-Host "=== QECS v1.3 Quantum Currency Initialization ==="
Write-Host "[INIT] Launching QECS Full Integration & Dashboard Verification"

# Initialize HARU
Write-Host "[STEP 1] Initializing HARU dynamic feedback learning"
python haru/autoregression.py --init --cycles 150 --verify_lambda_convergence
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] HARU initialization failed"
    exit 1
}

# Initialize Coherence Engine (HSMF + Phi-damping)
Write-Host "[STEP 2] Initializing Coherence Engine with phi-damping"
python src/core/coherence_engine.py --mode initialize --phi-damping
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Coherence engine initialization failed"
    exit 1
}

# Start Adaptive Gating Service
Write-Host "[STEP 3] Initializing Adaptive Gating Service"
python src/core/gating_service.py --adaptive --verify-safemode
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Gating service initialization failed"
    exit 1
}

# Sync Memory Subsystem
Write-Host "[STEP 4] Synchronizing Memory Subsystem"
python src/core/memory.py --sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Memory sync failed"
    exit 1
}

# Generate initial QRA for all nodes/users
Write-Host "[STEP 5] Generating QRA keys for all nodes"
python qra/generator.py --generate_all_nodes
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] QRA generation failed"
    exit 1
}
Write-Host "[QRA] Generated bioresonant QRA keys for all nodes"

# Start Continuous Attunement Daemon
Write-Host "[STEP 6] Starting Continuous Attunement Daemon"
Start-Process powershell -ArgumentList "-File start_continuous_attunement.ps1" -WindowStyle Hidden
Write-Host "[DAEMON] Continuous attunement running in background"

Write-Host "[INIT] QECS v1.3 Deployment Complete"
Write-Host ""
Write-Host "Access the dashboard at: http://localhost:8000/quantum_currency_dashboard.html"
Write-Host "API endpoints available at: http://localhost:5000"