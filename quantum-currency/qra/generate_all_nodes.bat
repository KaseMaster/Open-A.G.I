@echo off
REM generate_all_nodes.bat
REM Generate QRA keys for all nodes in the quantum currency system

echo [QRA] Generating bioresonant QRA keys for all nodes
python qra/generator.py --generate_all_nodes
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to generate QRA keys
    exit /b 1
)
echo [QRA] Generated bioresonant QRA keys for all nodes