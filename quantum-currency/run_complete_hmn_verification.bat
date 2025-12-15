@echo off
REM Complete HMN Production Coherence Verification & Stabilization Test
REM This script runs the comprehensive verification of the Harmonic Mesh Network

set PROJECT_DIR=D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency
set LOG_DIR=D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency\logs

echo 🚀 HMN Production Coherence Verification & Stabilization Test
echo =============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python is installed

REM Check if Waitress is available
python -c "import waitress" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Waitress not found. Installing...
    pip install waitress
    if errorlevel 1 (
        echo ❌ Failed to install Waitress
        pause
        exit /b 1
    )
)

echo ✅ Waitress is available

REM Create log directory
echo 📝 Setting up log directory...
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ✅ Log directory created

REM Run the verification script
echo 🔍 Running HMN verification...
echo.

cd /d "%PROJECT_DIR%"
python run_final_hmn_verification.py

if errorlevel 1 (
    echo ❌ Verification failed
    pause
    exit /b 1
) else (
    echo ✅ Verification completed successfully
)

echo.
echo =============================================================
echo 🎉 HMN Verification Process Complete!
echo =============================================================
echo.
echo 📋 Check the reports directory for detailed results
echo.
pause