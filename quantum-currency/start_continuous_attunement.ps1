# start_continuous_attunement.ps1
# Continuous background stabilizer ensuring Φ-harmonic feedback

Write-Host "[LAUNCH] Continuous Attunement Daemon Active"

while ($true) {
  Write-Host "[CYCLE] Running attunement cycle at $(Get-Date)"
  python haru/autoregression.py --update
  python src/core/stability.py --recalibrate
  python src/core/memory.py --sync
  Write-Host "[SLEEP] Sleeping for 5 seconds..."
  Start-Sleep -Seconds 5
}