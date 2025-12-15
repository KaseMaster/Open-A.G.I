@echo off
REM start_api_server.bat
REM Start the Quantum Currency Main API Server

echo ⚛️ Starting Quantum Currency Main API Server
echo.

cd src\api
python main_api.py

echo.
echo API server stopped.