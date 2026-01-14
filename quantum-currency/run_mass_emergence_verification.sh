#!/bin/bash

# Mass Emergence Verification Script for Unix/Linux
# This script runs the Mass Emergence verification and auto-tuning test

echo "===================================================="
echo "Quantum Currency System - Mass Emergence Verification"
echo "===================================================="

echo ""
echo "🧪 Starting Mass Emergence Verification Process..."
echo ""

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source ../.venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

# Run the verification script
echo "🚀 Executing Mass Emergence Verification..."
python3 "$(dirname "$0")/run_mass_emergence_verification.py"

# Check the exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Mass Emergence Verification completed successfully!"
    echo "📊 Check the reports directory for detailed results"
else
    echo ""
    echo "❌ Mass Emergence Verification failed"
    echo "📋 Please check the console output above for details"
fi

echo ""
echo "===================================================="
echo "Process completed at $(date)"
echo "===================================================="