#!/bin/bash
# start_continuous_attunement.sh
# Continuous background stabilizer ensuring Φ-harmonic feedback

echo "[LAUNCH] Continuous Attunement Daemon Active"
while true; do
  echo "[CYCLE] Running attunement cycle at $(date)"
  python3 haru/autoregression.py --update
  python3 src/core/stability.py --recalibrate
  python3 src/core/memory.py --sync
  echo "[SLEEP] Sleeping for 5 seconds..."
  sleep 5
done