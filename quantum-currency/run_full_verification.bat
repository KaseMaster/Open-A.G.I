@echo off
echo [Quantum Verification] Starting full production verification...

REM Start services
REM net start quantum-gunicorn
REM Healing timer may be scheduled as a Task Scheduler task
timeout /t 30

REM Launch epoch loop
python "%~dp0run_full_verification.py"

echo [Quantum Verification] Verification loop running. JSON reports at C:\opt\quantum-currency\logs\verification_report.json
pause