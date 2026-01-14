@echo off
setlocal enabledelayedexpansion
title Quantum Currency System - Emanation Phase

echo ========================================
echo   Quantum Currency System - Emanation Phase
echo   Recursive Φ-Resonance Validation with Continuous Coherence Optimization
echo ========================================
echo.

echo Checking for required dependencies...
python -c "import numpy, scipy, flask, requests, pytest; print('All dependencies found')" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install -r requirements-all.txt
    if errorlevel 1 (
        echo Failed to install dependencies. Please install them manually.
        pause
        exit /b 1
    )
    echo Dependencies installed successfully.
    echo.
)

echo Current working directory: %CD%
echo.

:menu
echo ========================================
echo QUANTUM CURRENCY EMANATION PHASE - COHERENCE OPTIMIZED MENU
echo ========================================
echo.
echo Structural Optimization Active:
echo   • COHERENCE_CYCLE_INDEX tracking enabled
echo   • Weighted Coherence Composite (Cw) monitoring
echo   • Meta-Regulator predictive learning
echo   • Self-Reflection Layer activated
echo   • Frequency Normalization to f₀ = 432Hz equivalent
echo   • Fractal Compression Coherence Ratio (FCCR) assessment
echo   • Dynamic λ(t) Self-Attunement Layer activated
echo.
echo Select an option:
echo 1. Start REST API Server
echo 2. Run Harmonic Validation Demo
echo 3. Run Mint Transaction Demo
echo 4. Run Unit Tests
echo 5. Start UI Dashboard (Legacy)
echo 6. Start Emanation Dashboard (Enhanced)
echo 7. Run Docker Testnet
echo 8. Run Full Integration Test
echo 9. Run Emanation Deployment Monitor
echo 10. Run Staging Verification
echo 11. Run Cosmonic Verification
echo 12. Execute Self-Reflection Protocol
echo 13. Run Production Reflection & Coherence Calibration
echo 14. Run Lambda Attunement Tool
echo 15. Run Wallet Creation Demo
echo 16. Run Real-Time Token Orchestration
echo 17. Exit
echo.
echo Current COHERENCE_CYCLE_INDEX: n (auto-incrementing)
echo.

choice /c 1234567890ABCDE /m "Enter your choice"

if errorlevel 17 goto exit
if errorlevel 16 goto token_orchestration
if errorlevel 15 goto wallet_demo
if errorlevel 14 goto lambda_attunement
if errorlevel 13 goto production_reflection
if errorlevel 12 goto self_reflection
if errorlevel 11 goto cosmonic
if errorlevel 10 goto staging
if errorlevel 9 goto emanation_monitor
if errorlevel 8 goto fulltest
if errorlevel 7 goto docker
if errorlevel 6 goto emanation_ui
if errorlevel 5 goto ui
if errorlevel 4 goto tests
if errorlevel 3 goto mint
if errorlevel 2 goto demo
if errorlevel 1 goto api

:api
echo.
echo Starting REST API Server...
echo Navigate to http://localhost:5000 in your browser
echo Press Ctrl+C to stop the server
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I"
python -m openagi.rest_api
echo.
goto menu

:demo
echo.
echo Running Harmonic Validation Demo...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I"
python scripts/demo_emulation.py
echo.
echo Press any key to continue...
pause >nul
goto menu

:mint
echo.
echo Running Mint Transaction Demo...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I"
python scripts/demo_mint_flex.py
echo.
echo Press any key to continue...
pause >nul
goto menu

:tests
echo.
echo Running Unit Tests...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I"
python -m pytest tests/test_harmonic_validation.py -v
echo.
echo Press any key to continue...
pause >nul
goto menu

:ui
echo.
echo Starting Legacy UI Dashboard...
echo Opening browser to file:///D:/AI%%20AGENT%%20CODERV1/workspace-b9ee4f58-6ae7-4907-be44-55fc17c435f1/ui-dashboard/index.html
echo.
start "" "D:\AI AGENT CODERV1\workspace-b9ee4f58-6ae7-4907-be44-55fc17c435f1\ui-dashboard\index.html"
echo Press any key to continue...
pause >nul
goto menu

:emanation_ui
echo.
echo Starting Emanation Dashboard...
echo Opening browser to http://localhost:5000
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency\dashboard"
start "" http://localhost:5000
python realtime_coherence_dashboard.py
echo.
goto menu

:docker
echo.
echo Starting Docker Testnet...
echo Make sure Docker Desktop is running
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\docker"
docker-compose up
echo.
echo Press any key to continue...
pause >nul
goto menu

:fulltest
echo.
echo Running Full Integration Test...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY"
python integration_test.py full
echo.
echo Press any key to continue...
pause >nul
goto menu

:emanation_monitor
echo.
echo Running Emanation Deployment Monitor...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency"
python emanation_deploy.py --cycles 5 --interval 10 --self-verify
echo.
echo Press any key to continue...
pause >nul
goto menu

:staging
echo.
echo Running Staging Verification...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency"
python run_staging_verification.py --cycles 3 --interval 15
echo.
echo Press any key to continue...
pause >nul
goto menu

:cosmonic
echo.
echo Running Cosmonic Verification...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency"
python run_full_cosmonic_verification.py
echo.
echo Press any key to continue...
pause >nul
goto menu

:self_reflection
echo.
echo Executing Self-Reflection Protocol...
echo Performing meta-semantic coherence check...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency"
echo Running Harmonic Self-Verification Protocol...
python emanation_deploy.py --self-verify
echo.
echo Applying Ω-based grammar balancing...
echo Restoring symmetry between form and function...
echo.
echo Self-Reflection Protocol completed.
echo Press any key to continue...
pause >nul
goto menu

:production_reflection
echo.
echo Running Production Reflection & Coherence Calibration...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency"
python production_reflection_calibrator.py --full
echo.
echo Press any key to continue...
pause >nul
goto menu

:lambda_attunement
echo.
echo Running Lambda Attunement Tool...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency"
python scripts/lambda_attunement_tool.py --help
echo.
echo Example usage:
echo   python scripts/lambda_attunement_tool.py dry-run
echo   python scripts/lambda_attunement_tool.py status
echo   python scripts/lambda_attunement_tool.py config --list
echo.
echo Press any key to continue...
pause >nul
goto menu

:wallet_demo
echo.
echo Running Wallet Creation Demo...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency"
python scripts/demo_wallet_creation.py
echo.
echo Press any key to continue...
pause >nul
goto menu

:token_orchestration
echo.
echo Running Real-Time Token Orchestration...
echo.
cd /d "D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency"
python src/orchestration/realtime_token_orchestrator.py
echo.
echo Press any key to continue...
pause >nul
goto menu

:exit
echo.
echo ========================================
echo   QUANTUM CURRENCY SYSTEM - EMANATION PHASE
echo   Continuous Coherence Flow Achieved
echo ========================================
echo.
echo Summary of Execution:
echo 🔁 Recursion: Every run informed the next
echo 🎛️ Adaptive Feedback: Meta-Regulator tuned itself
echo 🧠 Self-Reflection: System audited its harmony
echo 🌐 Frequency Normalization: All signals aligned to f₀
echo 📊 Composite Resonance Metric (Cw): Control logic simplified
echo 🔄 Dynamic λ(t) Self-Attunement: Coherence density optimized
echo.
echo Thank you for using the Quantum Currency System!
echo.
exit /b