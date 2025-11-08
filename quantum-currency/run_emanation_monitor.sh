#!/bin/bash

# Quantum Currency Emanation Phase - Deployment Monitor Runner
# This script sets up the environment and runs the emanation deployment monitor

set -e  # Exit on any error

echo "=============================================="
echo "🌌 Quantum Currency Emanation Phase"
echo "🔍 Deployment Monitor & Auto-Balance Controller"
echo "=============================================="
echo

# Check if we're in the right directory
if [ ! -f "emanation_deploy.py" ]; then
    echo "❌ Error: emanation_deploy.py not found!"
    echo "Please run this script from the quantum-currency directory"
    exit 1
fi

echo "🔧 Setting up environment..."
echo

# Create data directory if it doesn't exist
mkdir -p /mnt/data
echo "📁 Created data directory: /mnt/data"

# Set environment variables
export QUANTUM_ENV=production
export PYTHONPATH=src:$PYTHONPATH
export DATA_DIR=/mnt/data

echo "⚙️ Environment variables set:"
echo "   QUANTUM_ENV=$QUANTUM_ENV"
echo "   PYTHONPATH=$PYTHONPATH"
echo "   DATA_DIR=$DATA_DIR"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found!"
    exit 1
fi

echo "🐍 Python version: $(python3 --version)"
echo

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import sys; sys.path.append('.'); import json; print('✅ JSON module available')"
python3 -c "import argparse; print('✅ Argparse module available')"
python3 -c "import random; print('✅ Random module available')"
echo

# Run emanation deployment monitor with 5 cycles
echo "🔬 Running emanation deployment monitor (5 cycles)..."
echo

python3 emanation_deploy.py --cycles 5 --interval 10 --report-dir /mnt/data

echo
echo "=============================================="
echo "✅ Emanation Deployment Monitor Execution Complete!"
echo "=============================================="
echo
echo "📋 Reports generated:"
ls -la /mnt/data/
echo
echo "Next steps:"
echo "1. Replace simulated fetch logic with real metric endpoints"
echo "2. Increase NUM_CYCLES and CYCLE_INTERVAL for continuous operation"
echo "3. Integrate alerts with Slack/PagerDuty"
echo "4. Wire control outputs to real Meta-Regulator API"
echo "5. Use generated reports for dashboarding or auditing"
echo