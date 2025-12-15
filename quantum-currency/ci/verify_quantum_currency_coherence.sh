#!/bin/bash
# ci/verify_quantum_currency_coherence.sh
# Quantum Currency Integration Verification Pipeline

echo "[VERIFY] Starting Quantum Currency Coherence Pipeline"

python ci/simulate_unattunement.py
if [ $? -ne 0 ]; then
    echo "[ERROR] simulate_unattunement.py failed"
    exit 1
fi

python ci/verify_gas_threshold.py
if [ $? -ne 0 ]; then
    echo "[ERROR] verify_gas_threshold.py failed"
    exit 1
fi

python ci/verify_metrics.py
if [ $? -ne 0 ]; then
    echo "[ERROR] verify_metrics.py failed"
    exit 1
fi

python ci/test_safe_mode.py
if [ $? -ne 0 ]; then
    echo "[ERROR] test_safe_mode.py failed"
    exit 1
fi

python ci/validate_heatmap.py
if [ $? -ne 0 ]; then
    echo "[ERROR] validate_heatmap.py failed"
    exit 1
fi

echo "✅ All Quantum Currency Coherence Tests Passed"