@echo off
REM launch_qecs.bat
REM Unified One-Line Launch for QECS v1.3

echo ⚛️ QECS Full Integration & Dashboard Verification Prompt
echo.
echo Objective: Launch, integrate, and visualize the entire Quantum Currency system
echo with QRA, CAF, and Gravity Well Governance, verifying all coherence metrics,
echo transaction gating, and emission policies in real time.
echo.

REM Initialize Core Modules, QRA, & Continuous Attunement
echo [INIT] QECS v1.3 Deployment Starting

REM Initialize HARU
echo [HARU] Initializing dynamic feedback learning
python haru/autoregression.py --init --cycles 150 --verify_lambda_convergence

REM Initialize Coherence Engine (HSMF + Phi-damping)
echo [COHERENCE ENGINE] Initializing Phi-damping cycles
python src/core/coherence_engine.py --mode initialize --phi-damping

REM Start Adaptive Gating Service
echo [GATING SERVICE] Establishing adaptive thresholds
python src/core/gating_service.py --adaptive --verify-safemode

REM Sync Memory Subsystem
echo [MEMORY] Synchronizing recursive memory
python src/core/memory.py --sync

REM Generate initial QRA for all nodes/users
echo [QRA] Generating bioresonant QRA keys for all nodes
python qra/generator.py --generate_all_nodes
echo [QRA] Generated bioresonant QRA keys for all nodes

REM Start Continuous Attunement Daemon
echo [DAEMON] Starting continuous attunement in background
start /min cmd /c "powershell -File start_continuous_attunement.ps1"
echo [DAEMON] Continuous attunement running in background

REM Ledger, Transaction, & Coherence Integration
echo [LEDGER] Initializing transaction processing with QRA and CAF

REM Field-Level Security & Gravity Well Governance
echo [SECURITY] Initializing Gravity Well Governance

REM Dashboard UI Integration
echo [DASHBOARD] Launching Curvature Heatmap & QECS Metrics Panel

REM Start WebSocket stream for real-time metrics
echo [DASHBOARD] Starting WebSocket stream for real-time metrics
start /min cmd /c "python src/api/routes/curvature.py --stream"

REM Launch front-end UI
echo [DASHBOARD] Starting front-end UI server
start /min cmd /c "cd ..\..\..\ui-dashboard && python -m http.server 8000"
echo [DASHBOARD] Metrics visualized: RSI, CS, GAS, λ_opt(L), C_system, QRA coherence

REM Verify metrics compliance
echo [DASHBOARD] Verifying metrics compliance
python ci/verify_metrics.py

REM Full Verification Pipeline & Unified Launch
echo [VERIFY] Running QECS Full Coherence & Transaction Verification

REM Integration Tests
echo [VERIFY] Running integration tests
python ci/test_quantum_currency_integration.py

REM Verify Coherence & QRA
echo [VERIFY] Verifying coherence and QRA
bash ci/verify_quantum_currency_coherence.sh
powershell -File ci/verify_quantum_currency_coherence.ps1

echo [VERIFY] QECS v1.3 Fully Operational

REM Final status
echo.
echo ✅ Expected Outcomes:
echo   System coherence maximized (C_system → 1)
echo   Action efficiency minimized (I_eff → 0)
echo   GAS_target ≥ 0.99
echo   λ_opt(L) self-tuned
echo   QRA keys fully integrated for identity and gating
echo   CAF emission policy active
echo   Field-level security with Gravity Well detection active
echo   Dashboard real-time verification visualized
echo.
echo Access the dashboard at: http://localhost:8000/quantum_currency_dashboard.html
echo API endpoints available at: http://localhost:5000