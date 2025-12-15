@echo off
REM ci/verify_quantum_currency_coherence.bat
REM Quantum Currency Integration Verification Pipeline (Windows)

echo [VERIFY] Starting Quantum Currency Coherence Pipeline

python ci/simulate_unattunement.py
if %errorlevel% neq 0 (
    echo [ERROR] simulate_unattunement.py failed
    exit /b 1
)

python ci/verify_gas_threshold.py
if %errorlevel% neq 0 (
    echo [ERROR] verify_gas_threshold.py failed
    exit /b 1
)

python ci/verify_metrics.py
if %errorlevel% neq 0 (
    echo [ERROR] verify_metrics.py failed
    exit /b 1
)

python ci/test_safe_mode.py
if %errorlevel% neq 0 (
    echo [ERROR] test_safe_mode.py failed
    exit /b 1
)

python ci/validate_heatmap.py
if %errorlevel% neq 0 (
    echo [ERROR] validate_heatmap.py failed
    exit /b 1
)

echo ✅ All Quantum Currency Coherence Tests Passed