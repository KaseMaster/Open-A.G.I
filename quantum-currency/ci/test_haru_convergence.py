#!/usr/bin/env python3
"""
Test HARU (Harmonic Autoregression Unit) convergence
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_haru_initialization():
    """Test HARU initialization and convergence"""
    try:
        from haru.autoregression import HARU
        
        # Initialize HARU
        haru = HARU(cycles=50)
        
        # Simulate initialization cycles
        for i in range(50):
            # Mock state data for initialization
            current_state = {
                'gas': 0.95 + np.random.normal(0, 0.02),
                'cs': 0.9 + np.random.normal(0, 0.01),
                'rsi': 0.7 + np.random.normal(0, 0.05),
                'phi_ratio_deviation': 0.005 + np.random.normal(0, 0.001)
            }
            
            history = {}  # Empty history for initialization
            haru.update(current_state, history)
        
        # Check convergence
        converged = haru.verify_lambda_convergence()
        
        if converged:
            print("✅ HARU convergence test PASSED")
            return True
        else:
            print("❌ HARU convergence test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ HARU test failed with exception: {e}")
        return False

def test_hsmf_computation():
    """Test HSMF computation"""
    try:
        from hsmf import HarmonicComputationalFramework, C_CRIT
        
        # Initialize framework
        hsmf = HarmonicComputationalFramework()
        
        # Example state
        current_state = {
            'gas': 0.96,
            'cs': 0.91,
            'rsi': 0.68,
            'phi_ratio_deviation': 0.008,
            'target_gas': 0.95
        }
        
        history = {}
        
        action_proposals = {
            'actions': ['tx1', 'tx2'],
            'resource_cost': 0.8,
            'lambda1': 0.1,
            'lambda2': 0.2
        }
        
        # Execute computational cycle
        result = hsmf.phi_damping_computational_cycle(current_state, history, action_proposals)
        
        # Verify results
        assert result['C_system'] >= 0.0, "C_system should be non-negative"
        assert result['GAS'] >= 0.0, "GAS should be non-negative"
        assert result['RSI'] >= 0.0, "RSI should be non-negative"
        
        print("✅ HSMF computation test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ HSMF computation test failed with exception: {e}")
        return False

def main():
    """Run all tests"""
    print("[TEST] Running HARU and HSMF tests...")
    
    test1_passed = test_haru_initialization()
    test2_passed = test_hsmf_computation()
    
    if test1_passed and test2_passed:
        print("✅ All tests PASSED")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()