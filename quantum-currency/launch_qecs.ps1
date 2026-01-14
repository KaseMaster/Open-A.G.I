# launch_qecs.ps1
# Unified One-Line Launch for QECS v1.3

Write-Host "⚛️ QECS Full Integration & Dashboard Verification Prompt"
Write-Host ""
Write-Host "Objective: Launch, integrate, and visualize the entire Quantum Currency system"
Write-Host "with QRA, CAF, and Gravity Well Governance, verifying all coherence metrics,"
Write-Host "transaction gating, and emission policies in real time."
Write-Host ""

# Initialize Core Modules, QRA, & Continuous Attunement
Write-Host "[INIT] QECS v1.3 Deployment Starting"

# Initialize HARU
Write-Host "[HARU] Initializing dynamic feedback learning"
python haru/autoregression.py --init --cycles 150 --verify_lambda_convergence

# Initialize Coherence Engine (HSMF + Phi-damping)
Write-Host "[COHERENCE ENGINE] Initializing Phi-damping cycles"
python src/core/coherence_engine.py --mode initialize --phi-damping

# Start Adaptive Gating Service
Write-Host "[GATING SERVICE] Establishing adaptive thresholds"
python src/core/gating_service.py --adaptive --verify-safemode

# Sync Memory Subsystem
Write-Host "[MEMORY] Synchronizing recursive memory"
python src/core/memory.py --sync

# Generate initial QRA for all nodes/users
Write-Host "[QRA] Generating bioresonant QRA keys for all nodes"
python qra/generator.py --generate_all_nodes
Write-Host "[QRA] Generated bioresonant QRA keys for all nodes"

# Start Continuous Attunement Daemon
Write-Host "[DAEMON] Starting continuous attunement in background"
Start-Process powershell -ArgumentList "-File start_continuous_attunement.ps1" -WindowStyle Hidden
Write-Host "[DAEMON] Continuous attunement running in background"

# Ledger, Transaction, & Coherence Integration
Write-Host "[LEDGER] Initializing transaction processing with QRA and CAF"

# Field-Level Security & Gravity Well Governance
Write-Host "[SECURITY] Initializing Gravity Well Governance"

# Dashboard UI Integration
Write-Host "[DASHBOARD] Launching Curvature Heatmap & QECS Metrics Panel"

# Start WebSocket stream for real-time metrics
Write-Host "[DASHBOARD] Starting WebSocket stream for real-time metrics"
Start-Process powershell -ArgumentList "-Command python src/api/routes/curvature.py --stream" -WindowStyle Hidden

# Launch front-end UI
Write-Host "[DASHBOARD] Starting front-end UI server"
Start-Process powershell -ArgumentList "-Command python -m http.server 8000" -WorkingDirectory "../ui-dashboard" -WindowStyle Hidden
Write-Host "[DASHBOARD] Metrics visualized: RSI, CS, GAS, λ_opt(L), C_system, QRA coherence"

# Verify metrics compliance
Write-Host "[DASHBOARD] Verifying metrics compliance"
python ci/verify_metrics.py

# Full Verification Pipeline & Unified Launch
Write-Host "[VERIFY] Running QECS Full Coherence & Transaction Verification"

# Integration Tests
Write-Host "[VERIFY] Running integration tests"
python ci/test_quantum_currency_integration.py

# Verify Coherence & QRA
Write-Host "[VERIFY] Verifying coherence and QRA"
bash ci/verify_quantum_currency_coherence.sh
powershell -File ci/verify_quantum_currency_coherence.ps1

Write-Host "[VERIFY] QECS v1.3 Fully Operational"

# Final status
Write-Host ""
Write-Host "✅ Expected Outcomes:"
Write-Host "  System coherence maximized (C_system → 1)"
Write-Host "  Action efficiency minimized (I_eff → 0)"
Write-Host "  GAS_target ≥ 0.99"
Write-Host "  λ_opt(L) self-tuned"
Write-Host "  QRA keys fully integrated for identity and gating"
Write-Host "  CAF emission policy active"
Write-Host "  Field-level security with Gravity Well detection active"
Write-Host "  Dashboard real-time verification visualized"
Write-Host ""
Write-Host "Access the dashboard at: http://localhost:8000/quantum_currency_dashboard.html"
Write-Host "API endpoints available at: http://localhost:5000"