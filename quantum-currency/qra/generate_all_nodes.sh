#!/bin/bash
# generate_all_nodes.sh
# Generate QRA keys for all nodes in the quantum currency system

echo "[QRA] Generating bioresonant QRA keys for all nodes"
python3 qra/generator.py --generate_all_nodes
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to generate QRA keys"
    exit 1
fi
echo "[QRA] Generated bioresonant QRA keys for all nodes"