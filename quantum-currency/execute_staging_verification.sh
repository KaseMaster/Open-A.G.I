#!/bin/bash

# Quantum Currency Emanation Phase - Staging Verification Executor
# This script sets up the environment and runs the staging verification process

set -e  # Exit on any error

echo "=============================================="
echo "🚀 Quantum Currency Emanation Phase"
echo "🔍 Staging Verification Executor"
echo "=============================================="
echo

# Check if we're in the right directory
if [ ! -f "run_staging_verification.py" ]; then
    echo "❌ Error: run_staging_verification.py not found!"
    echo "Please run this script from the quantum-currency directory"
    exit 1
fi

echo "🔧 Setting up environment..."
echo

# Create reports directory if it doesn't exist
mkdir -p reports/staging
echo "📁 Created reports directory: reports/staging"

# Set environment variables
export QUANTUM_ENV=staging
export PYTHONPATH=src:$PYTHONPATH
export REPORT_OUTPUT_DIR=reports/staging

echo "⚙️ Environment variables set:"
echo "   QUANTUM_ENV=$QUANTUM_ENV"
echo "   PYTHONPATH=$PYTHONPATH"
echo "   REPORT_OUTPUT_DIR=$REPORT_OUTPUT_DIR"
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
python3 -c "import sys; sys.path.append('.'); from src.core.cosmonic_verification import CosmonicVerificationSystem; print('✅ Cosmonic Verification System available')"
python3 -c "import json; print('✅ JSON module available')"
python3 -c "import argparse; print('✅ Argparse module available')"
echo

# Run staging verification with 3 cycles
echo "🔬 Running staging verification (3 cycles)..."
echo

python3 run_staging_verification.py --cycles 3 --interval 15 --output reports/staging/final_verification_report.json

echo
echo "=============================================="
echo "✅ Staging Verification Execution Complete!"
echo "=============================================="
echo
echo "📋 Reports generated:"
ls -la reports/staging/
echo
echo "Next steps:"
echo "1. Review the verification reports"
echo "2. Address any failed components"
echo "3. Run performance tests"
echo "4. Execute security audits"
echo "5. Proceed to production deployment when ready"
echo