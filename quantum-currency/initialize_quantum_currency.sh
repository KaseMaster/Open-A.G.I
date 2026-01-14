#!/bin/bash
# initialize_quantum_currency.sh
# GHSP v1.2 Quantum Currency Initialization

echo "=== GHSP v1.2 Quantum Currency Initialization ==="
echo "[INIT] Launching Quantum Currency Integration Directive (QCI–HSMF v1.2)"

# Activate HARU dynamic feedback learning
echo "[STEP 1] Activating HARU dynamic feedback learning"
python3 haru/autoregression.py --init --cycles 150 --verify_lambda_convergence
if [ $? -ne 0 ]; then
    echo "[ERROR] HARU initialization failed"
    exit 1
fi

# Establish field-level coherence engine
echo "[STEP 2] Establishing field-level coherence engine"
python3 src/core/coherence_engine.py --mode initialize --phi-damping
if [ $? -ne 0 ]; then
    echo "[ERROR] Coherence engine initialization failed"
    exit 1
fi

# Initialize the gating service with adaptive thresholds
echo "[STEP 3] Initializing gating service with adaptive thresholds"
python3 src/core/gating_service.py --adaptive --verify-safemode
if [ $? -ne 0 ]; then
    echo "[ERROR] Gating service initialization failed"
    exit 1
fi

# Sync recursive coherence memory and heatmap dashboard
echo "[STEP 4] Syncing recursive coherence memory"
python3 src/core/memory.py --sync
if [ $? -ne 0 ]; then
    echo "[ERROR] Memory sync failed"
    exit 1
fi

echo "[STEP 5] Streaming curvature dashboard"
python3 src/api/routes/curvature.py --stream --verify
if [ $? -ne 0 ]; then
    echo "[ERROR] Curvature dashboard streaming failed"
    exit 1
fi

# Start full continuous attunement verification cycle
echo "[STEP 6] Starting full continuous attunement verification cycle"
bash ci/verify_quantum_currency_coherence.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] Verification pipeline failed"
    exit 1
fi

echo "✅ GHSP v1.2 Continuous Attunement Active"