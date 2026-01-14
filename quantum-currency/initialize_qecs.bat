@echo off
REM initialize_qecs.bat
REM QECS v1.3 Quantum Currency Initialization

echo === QECS v1.3 Quantum Currency Initialization ===
echo [INIT] Launching QECS Full Integration & Dashboard Verification

REM Initialize HARU
echo [STEP 1] Initializing HARU dynamic feedback learning
python haru/autoregression.py --init --cycles 150 --verify_lambda_convergence
if %ERRORLEVEL% neq 0 (
    echo [ERROR] HARU initialization failed
    exit /b 1
)

REM Initialize Coherence Engine (HSMF + Phi-damping)
echo [STEP 2] Initializing Coherence Engine with phi-damping
python src/core/coherence_engine.py --mode initialize --phi-damping
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Coherence engine initialization failed
    exit /b 1
)

REM Start Adaptive Gating Service
echo [STEP 3] Initializing Adaptive Gating Service
python src/core/gating_service.py --adaptive --verify-safemode
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Gating service initialization failed
    exit /b 1
)

REM Sync Memory Subsystem
echo [STEP 4] Synchronizing Memory Subsystem
python src/core/memory.py --sync
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Memory sync failed
    exit /b 1
)

REM Generate initial QRA for all nodes/users
echo [STEP 5] Generating QRA keys for all nodes
python qra/generator.py --generate_all_nodes
if %ERRORLEVEL% neq 0 (
    echo [ERROR] QRA generation failed
    exit /b 1
)
echo [QRA] Generated bioresonant QRA keys for all nodes

REM Start Continuous Attunement Daemon
echo [STEP 6] Starting Continuous Attunement Daemon
start /min cmd /c "powershell -File start_continuous_attunement.ps1"
echo [DAEMON] Continuous attunement running in background

echo [INIT] QECS v1.3 Deployment Complete
echo.
echo Access the dashboard at: http://localhost:8000/quantum_currency_dashboard.html
echo API endpoints available at: http://localhost:5000