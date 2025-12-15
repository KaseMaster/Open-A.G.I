@echo off
REM start_continuous_attunement.bat
REM Continuous background stabilizer ensuring Φ-harmonic feedback

echo [LAUNCH] Continuous Attunement Daemon Active
:loop
echo [CYCLE] Running attunement cycle at %date% %time%
python haru/autoregression.py --update
python src/core/stability.py --recalibrate
python src/core/memory.py --sync
echo [SLEEP] Sleeping for 5 seconds...
timeout /t 5 /nobreak >nul
goto loop