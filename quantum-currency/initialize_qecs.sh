#!/bin/bash
# initialize_qecs.sh
# QECS v1.3 Quantum Currency Initialization

echo "=== QECS v1.3 Quantum Currency Initialization ==="
echo "[INIT] Launching QECS Full Integration & Dashboard Verification"

# Initialize HARU
echo "[STEP 1] Initializing HARU dynamic feedback learning"
python3 haru/autoregression.py --init --cycles 150 --verify_lambda_convergence
if [ $? -ne 0 ]; then
    echo "[ERROR] HARU initialization failed"
    exit 1
fi

# Initialize Coherence Engine (HSMF + Phi-damping)
echo "[STEP 2] Initializing Coherence Engine with phi-damping"
python3 src/core/coherence_engine.py --mode initialize --phi-damping
if [ $? -ne 0 ]; then
    echo "[ERROR] Coherence engine initialization failed"
    exit 1
fi

# Start Adaptive Gating Service
echo "[STEP 3] Initializing Adaptive Gating Service"
python3 src/core/gating_service.py --adaptive --verify-safemode
if [ $? -ne 0 ]; then
    echo "[ERROR] Gating service initialization failed"
    exit 1
fi

# Sync Memory Subsystem
echo "[STEP 4] Synchronizing Memory Subsystem"
python3 src/core/memory.py --sync
if [ $? -ne 0 ]; then
    echo "[ERROR] Memory sync failed"
    exit 1
fi

# Generate initial QRA for all nodes/users
echo "[STEP 5] Generating QRA keys for all nodes"
python3 qra/generator.py --generate_all_nodes
if [ $? -ne 0 ]; then
    echo "[ERROR] QRA generation failed"
    exit 1
fi
echo "[QRA] Generated bioresonant QRA keys for all nodes"

# Start Continuous Attunement Daemon
echo "[STEP 6] Starting Continuous Attunement Daemon"
bash start_continuous_attunement.sh &
echo "[DAEMON] Continuous attunement running in background"

echo "[INIT] QECS v1.3 Deployment Complete"
echo ""
echo "Access the dashboard at: http://localhost:8000/quantum_currency_dashboard.html"
echo "API endpoints available at: http://localhost:5000"