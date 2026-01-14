@echo off
REM initialize_quantum_currency.bat
REM GHSP v1.2 Quantum Currency Initialization

echo === GHSP v1.2 Quantum Currency Initialization ===
echo [INIT] Launching Quantum Currency Integration Directive (QCI–HSMF v1.2)

REM Activate HARU dynamic feedback learning
echo [STEP 1] Activating HARU dynamic feedback learning
python haru/autoregression.py --init --cycles 150 --verify_lambda_convergence
if %errorlevel% neq 0 (
    echo [ERROR] HARU initialization failed
    exit /b 1
)

REM Establish field-level coherence engine
echo [STEP 2] Establishing field-level coherence engine
python src/core/coherence_engine.py --mode initialize --phi-damping
if %errorlevel% neq 0 (
    echo [ERROR] Coherence engine initialization failed
    exit /b 1
)

REM Initialize the gating service with adaptive thresholds
echo [STEP 3] Initializing gating service with adaptive thresholds
python src/core/gating_service.py --adaptive --verify-safemode
if %errorlevel% neq 0 (
    echo [ERROR] Gating service initialization failed
    exit /b 1
)

REM Sync recursive coherence memory and heatmap dashboard
echo [STEP 4] Syncing recursive coherence memory
python src/core/memory.py --sync
if %errorlevel% neq 0 (
    echo [ERROR] Memory sync failed
    exit /b 1
)

echo [STEP 5] Streaming curvature dashboard
python src/api/routes/curvature.py --stream --verify
if %errorlevel% neq 0 (
    echo [ERROR] Curvature dashboard streaming failed
    exit /b 1
)

REM Start full continuous attunement verification cycle
echo [STEP 6] Starting full continuous attunement verification cycle
call ci/verify_quantum_currency_coherence.bat
if %errorlevel% neq 0 (
    echo [ERROR] Verification pipeline failed
    exit /b 1
)

echo ✅ GHSP v1.2 Continuous Attunement Active