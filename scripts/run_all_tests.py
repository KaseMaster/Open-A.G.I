#!/usr/bin/env python3
"""
Script to run all tests for the Quantum Currency System
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run all tests"""
    print("Running all tests for Quantum Currency System")
    
    # Change to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Run all tests
    test_commands = [
        "python -m pytest tests/test_consensus.py -v",
        "python -m pytest tests/test_ledger_api.py -v",
        "python -m pytest tests/test_coherence_cycle.py -v",
        "python -m pytest tests/test_end_to_end_integration.py -v",
        "python -m pytest tests/test_token_rules.py -v",
        "python -m pytest tests/test_harmonic_validation.py -v",
        "python -m pytest tests/test_token_coherence_integration.py -v"
    ]
    
    all_passed = True
    for command in test_commands:
        if not run_command(command):
            all_passed = False
    
    # Run coverage report (if coverage is available)
    print("\nRunning coverage report...")
    coverage_result = run_command("python -m pytest --cov=openagi --cov-report=html")
    if not coverage_result:
        print("Coverage report failed, but continuing with basic coverage...")
        run_command("python -m pytest --cov=openagi")
    
    if all_passed:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())