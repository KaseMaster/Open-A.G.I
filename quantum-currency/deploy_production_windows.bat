@echo off
REM Quantum Currency Production Deployment Script for Windows
REM This script demonstrates the deployment process on Windows systems

set PROJECT_NAME=quantum-currency
set PROJECT_DIR=D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency
set LOG_DIR=D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency\logs

echo 🚀 Quantum Currency Production Deployment for Windows
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python first.
    exit /b 1
)

echo ✅ Python is installed

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip first.
    exit /b 1
)

echo ✅ pip is installed

REM Install Python dependencies
echo 🐍 Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install Python dependencies
    exit /b 1
)

echo ✅ Python dependencies installed

REM Create log directory
echo 📝 Setting up log directory...
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ✅ Log directory created

REM Start the application using Gunicorn (if available) or Flask development server
echo 🚀 Starting Quantum Currency application...

REM Check if Gunicorn is available
gunicorn --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Gunicorn not found. Using Flask development server...
    echo 🐍 Starting Flask development server...
    python src/api/main.py
) else (
    echo ✅ Gunicorn found. Starting with Gunicorn...
    gunicorn --workers 4 --bind 0.0.0.0:5000 src.api.main:app
)

echo ✅ Quantum Currency application started

echo ==========================================
echo 🎉 Quantum Currency Production Deployment Complete!
echo ==========================================
echo 📝 Summary:
echo   - Project directory: %PROJECT_DIR%
echo   - Logs directory: %LOG_DIR%
echo   - Application running on: http://localhost:5000
echo.
echo 🔧 Health check endpoints:
echo   - http://localhost:5000/health
echo   - http://localhost:5000/metrics
echo.
echo 🔄 To restart the application, run this script again
echo.
echo 📋 To view logs, check the logs directory
echo.
echo 🛡️ Note: This is a simplified deployment for Windows.
echo    For production use, consider using Docker or WSL with Linux.
echo.
echo ✅ Deployment completed successfully!