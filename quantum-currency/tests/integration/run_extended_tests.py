#!/usr/bin/env python3
"""
Runner script for all extended test suites
"""

import subprocess
import sys
import os
from pathlib import Path

def run_test_suite(name, command):
    """Run a test suite and return success status"""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        success = result.returncode == 0
        status = "PASSED" if success else "FAILED"
        print(f"\n{status} {name}")
        return success
    except Exception as e:
        print(f"FAILED {name} - Exception: {e}")
        return False

def main():
    """Run all extended test suites"""
    print("Quantum Currency Extended Test Suite Runner")
    print("=" * 60)
    
    # Change to quantum-currency directory
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    
    test_suites = [
        ("Performance Tests", "python -m pytest tests/cal/test_cal_performance.py -v --benchmark-only --benchmark-sort=mean"),
        ("Observer Edge Case Tests", "python -m pytest tests/monitoring/test_observer_edge_cases.py -v"),
        ("Security Penetration Tests", "python -m pytest tests/security/test_omega_security_penetration.py -v"),
    ]
    
    results = []
    for name, command in test_suites:
        success = run_test_suite(name, command)
        results.append((name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Suite Summary")
    print(f"{'='*60}")
    
    passed = 0
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{status} {name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} test suites passed")
    
    # Exit with appropriate code
    sys.exit(0 if passed == len(results) else 1)

if __name__ == "__main__":
    main()