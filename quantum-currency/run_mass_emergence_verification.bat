@echo off
REM Mass Emergence Verification Script for Windows
REM This script runs the Mass Emergence verification and auto-tuning test

echo ====================================================
echo Quantum Currency System - Mass Emergence Verification
echo ====================================================

echo.
echo 🧪 Starting Mass Emergence Verification Process...
echo.

REM Activate virtual environment if it exists
if exist "..\.venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call ..\.venv\Scripts\activate.bat
) else (
    echo ⚠️  No virtual environment found. Using system Python.
)

REM Run the verification script
echo 🚀 Executing Mass Emergence Verification...
python "%~dp0run_mass_emergence_verification.py"

REM Check the exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Mass Emergence Verification completed successfully!
    echo 📊 Check the reports directory for detailed results
) else (
    echo.
    echo ❌ Mass Emergence Verification failed with error code %ERRORLEVEL%
    echo 📋 Please check the console output above for details
)

echo.
echo ====================================================
echo Process completed at %date% %time%
echo ====================================================

pause