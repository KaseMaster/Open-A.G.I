#!/usr/bin/env python3
"""
Verify GAS Threshold for CI Pipeline
Ensures Geometric Alignment Score meets requirements
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.gating_service import GatingService

def verify_gas_threshold():
    """Verify GAS threshold requirements"""
    print("🔍 Verifying GAS Threshold Requirements...")
    print("=" * 40)
    
    # Create gating service
    gating_service = GatingService()
    
    # Test cases for different stability levels
    test_cases = [
        {
            "name": "Stable Conditions",
            "R_Omega_magnitude": 0.95e-62,
            "coherence_stability": 0.95,
            "rsi": 0.90,
            "expected_gas_min": 0.70,
            "expected_gas_max": 0.75
        },
        {
            "name": "Globally Attuned",
            "R_Omega_magnitude": 2.5e-62,
            "coherence_stability": 0.98,
            "rsi": 0.95,
            "expected_gas_min": 0.99,
            "expected_gas_max": 1.00
        },
        {
            "name": "Marginal Stability",
            "R_Omega_magnitude": 0.85e-62,
            "coherence_stability": 0.85,
            "rsi": 0.80,
            "expected_gas_min": 0.60,
            "expected_gas_max": 0.70
        }
    ]
    
    test_results = []
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['name']}")
        
        # Create R_Ω tensor
        R_Omega = np.array([
            [case['R_Omega_magnitude'], 0, 0, 0],
            [0, case['R_Omega_magnitude'], 0, 0],
            [0, 0, case['R_Omega_magnitude'], 0],
            [0, 0, 0, case['R_Omega_magnitude']]
        ])
        
        eigenvalues = {'n': 3, 'l': 2, 'm': 1, 's': 0.5}
        
        # Calculate metrics
        metrics = gating_service.get_current_metrics(
            R_Omega_tensor=R_Omega,
            geometric_eigenvalues=eigenvalues,
            coherence_stability=case['coherence_stability'],
            rsi=case['rsi']
        )
        
        print(f"  GAS: {metrics.gas:.4f}")
        print(f"  CS: {metrics.cs:.4f}")
        print(f"  RSI: {metrics.rsi:.4f}")
        
        # Check if GAS is within expected range
        if case['expected_gas_min'] <= metrics.gas <= case['expected_gas_max']:
            print(f"  ✅ GAS within expected range ({case['expected_gas_min']:.2f} - {case['expected_gas_max']:.2f})")
            test_results.append(True)
        else:
            print(f"  ❌ GAS outside expected range ({case['expected_gas_min']:.2f} - {case['expected_gas_max']:.2f})")
            test_results.append(False)
    
    # Verify L_Φ writes only succeed when both CS > 0.80 and GAS > 0.99
    print(f"\n🔐 Testing L_Φ Write Conditions...")
    
    # Test case: Both conditions met
    print("  Test: CS > 0.80 AND GAS > 0.99")
    R_Omega = np.array([
        [2.5e-62, 0, 0, 0],
        [0, 2.5e-62, 0, 0],
        [0, 0, 2.5e-62, 0],
        [0, 0, 0, 2.5e-62]
    ])
    
    metrics = gating_service.get_current_metrics(
        R_Omega_tensor=R_Omega,
        geometric_eigenvalues={'n': 4, 'l': 3, 'm': 2, 's': 0.5},
        coherence_stability=0.95,  # > 0.80
        rsi=0.95
    )
    
    # Both conditions should be met for L_Φ writes
    cs_condition = metrics.cs > 0.80
    gas_condition = metrics.gas > 0.99
    
    if cs_condition and gas_condition:
        print(f"    CS: {metrics.cs:.4f} > 0.80 ✅")
        print(f"    GAS: {metrics.gas:.4f} > 0.99 ✅")
        print(f"    L_Φ writes should SUCCEED ✅")
        test_results.append(True)
    else:
        print(f"    CS: {metrics.cs:.4f} > 0.80 {'✅' if cs_condition else '❌'}")
        print(f"    GAS: {metrics.gas:.4f} > 0.99 {'✅' if gas_condition else '❌'}")
        print(f"    L_Φ writes should FAIL ❌")
        test_results.append(False)
    
    # Overall result
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 GAS Threshold Verification: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All GAS threshold tests PASSED")
        return True
    else:
        print("💥 Some GAS threshold tests FAILED")
        return False

if __name__ == "__main__":
    success = verify_gas_threshold()
    sys.exit(0 if success else 1)