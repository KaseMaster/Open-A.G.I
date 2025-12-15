@echo off
REM verify_qecs.bat
REM QECS Full Coherence & Transaction Verification

echo [VERIFY] Running QECS Full Coherence & Transaction Verification

REM Integration Tests
echo [STEP 1] Running Integration Tests
python ci/test_quantum_currency_integration.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Integration tests failed
    exit /b 1
)

REM Verify Coherence & QRA
echo [STEP 2] Verifying Coherence & QRA
bash ci/verify_quantum_currency_coherence.sh
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Coherence verification failed
    exit /b 1
)

REM Windows support
echo [STEP 3] Running Windows verification
powershell -File ci/verify_quantum_currency_coherence.ps1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Windows verification failed
    exit /b 1
)

echo [VERIFY] QECS v1.3 Fully Operational
echo.
echo Expected Outcomes:
echo   System coherence maximized (C_system → 1)
echo   Action efficiency minimized (I_eff → 0)
echo   GAS_target ≥ 0.99
echo   λ_opt(L) self-tuned
echo   QRA keys fully integrated for identity and gating
echo   CAF emission policy active
echo   Field-level security with Gravity Well detection active
echo   Dashboard real-time verification visualized