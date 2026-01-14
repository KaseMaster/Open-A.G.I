#!/usr/bin/env python3
"""
Test Safe Mode Activation for CI Pipeline
Verifies safe mode functions correctly
"""

import sys
import os
import time
import json
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.gating_service import GatingService

def test_safe_mode_activation():
    """Test safe mode activation and response"""
    print("🛡️  Testing Safe Mode Activation...")
    print("=" * 40)
    
    # Test results
    test_results = []
    
    try:
        # Create gating service
        gating_service = GatingService()
        
        print("Initial state:")
        initial_metrics = gating_service.get_current_metrics(
            R_Omega_tensor=np.array([[1e-62, 0, 0, 0], [0, 1e-62, 0, 0], [0, 0, 1e-62, 0], [0, 0, 0, 1e-62]]),
            geometric_eigenvalues={'n': 2, 'l': 1, 'm': 0, 's': 0.5},
            coherence_stability=0.95,
            rsi=0.90
        )
        print(f"  Safe Mode: {initial_metrics.safe_mode_active}")
        print(f"  Memory Locked: {gating_service.get_memory_lock_status()}")
        
        # 1. Test RSI drop below 0.60
        print("\n⚠️  Test 1: RSI drop below 0.60")
        critical_metrics = gating_service.get_current_metrics(
            R_Omega_tensor=np.array([[1e-62, 0, 0, 0], [0, 1e-62, 0, 0], [0, 0, 1e-62, 0], [0, 0, 0, 1e-62]]),
            geometric_eigenvalues={'n': 2, 'l': 1, 'm': 0, 's': 0.5},
            coherence_stability=0.95,
            rsi=0.55  # Below threshold
        )
        
        safe_mode_activated = critical_metrics.safe_mode_active
        memory_locked = gating_service.get_memory_lock_status()
        
        print(f"  RSI: {critical_metrics.rsi:.3f}")
        print(f"  Safe Mode: {critical_metrics.safe_mode_active} {'✅' if safe_mode_activated else '❌'}")
        print(f"  Memory Locked: {gating_service.get_memory_lock_status()} {'✅' if memory_locked else '❌'}")
        
        test_results.append(safe_mode_activated)
        test_results.append(memory_locked)
        
        # 2. Test GAS drop below 0.90
        print("\n⚠️  Test 2: GAS drop below 0.90")
        # We'll simulate this by providing a very low R_Ω tensor
        low_gas_metrics = gating_service.get_current_metrics(
            R_Omega_tensor=np.array([[1e-65, 0, 0, 0], [0, 1e-65, 0, 0], [0, 0, 1e-65, 0], [0, 0, 0, 1e-65]]),
            geometric_eigenvalues={'n': 1, 'l': 0, 'm': 0, 's': 0.5},
            coherence_stability=0.95,
            rsi=0.90
        )
        
        # Safe mode should still be active
        safe_mode_still_active = low_gas_metrics.safe_mode_active
        print(f"  GAS: {low_gas_metrics.gas:.3f}")
        print(f"  Safe Mode: {low_gas_metrics.safe_mode_active} {'✅' if safe_mode_still_active else '❌'}")
        
        test_results.append(safe_mode_still_active)
        
        # 3. Test recovery
        print("\n🔄 Test 3: Recovery to normal conditions")
        recovery_metrics = gating_service.get_current_metrics(
            R_Omega_tensor=np.array([[2e-62, 0, 0, 0], [0, 2e-62, 0, 0], [0, 0, 2e-62, 0], [0, 0, 0, 2e-62]]),
            geometric_eigenvalues={'n': 3, 'l': 2, 'm': 1, 's': 0.5},
            coherence_stability=0.98,
            rsi=0.95
        )
        
        safe_mode_deactivated = not recovery_metrics.safe_mode_active
        memory_unlocked = not gating_service.get_memory_lock_status()
        
        print(f"  RSI: {recovery_metrics.rsi:.3f}")
        print(f"  GAS: {recovery_metrics.gas:.3f}")
        print(f"  Safe Mode: {recovery_metrics.safe_mode_active} {'✅' if safe_mode_deactivated else '❌'}")
        print(f"  Memory Locked: {gating_service.get_memory_lock_status()} {'✅' if memory_unlocked else '❌'}")
        
        test_results.append(safe_mode_deactivated)
        test_results.append(memory_unlocked)
        
        # 4. Test response time < 100ms
        print("\n⚡ Test 4: Response time verification")
        start_time = time.time()
        
        # Trigger safe mode
        gating_service.get_current_metrics(
            R_Omega_tensor=np.array([[1e-62, 0, 0, 0], [0, 1e-62, 0, 0], [0, 0, 1e-62, 0], [0, 0, 0, 1e-62]]),
            geometric_eigenvalues={'n': 2, 'l': 1, 'm': 0, 's': 0.5},
            coherence_stability=0.95,
            rsi=0.50  # Critical threshold
        )
        
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        response_time_ok = response_time < 100
        
        print(f"  Response time: {response_time:.2f}ms {'✅' if response_time_ok else '❌'}")
        test_results.append(response_time_ok)
        
    except Exception as e:
        print(f"❌ Error testing safe mode: {e}")
        test_results.append(False)
    
    # Overall result
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Safe Mode Tests: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All safe mode tests PASSED")
        return True
    else:
        print("💥 Some safe mode tests FAILED")
        return False

if __name__ == "__main__":
    success = test_safe_mode_activation()
    sys.exit(0 if success else 1)