#!/usr/bin/env python3
"""
Simulate Unattunement Event for CI Pipeline
Tests the gating service response to critical conditions
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.gating_service import GatingService

def simulate_unattunement_event():
    """Simulate an unattunement event and verify system response"""
    print("🔬 Simulating Unattunement Event...")
    print("=" * 40)
    
    # Create gating service
    gating_service = GatingService()
    
    # Initial stable state
    print("Initial state - Stable conditions:")
    R_Omega = np.array([
        [2.0e-62, 0, 0, 0],
        [0, 2.0e-62, 0, 0],
        [0, 0, 2.0e-62, 0],
        [0, 0, 0, 2.0e-62]
    ])
    
    eigenvalues = {'n': 3, 'l': 2, 'm': 1, 's': 0.5}
    
    # Get initial metrics
    metrics = gating_service.get_current_metrics(
        R_Omega_tensor=R_Omega,
        geometric_eigenvalues=eigenvalues,
        coherence_stability=0.95,
        rsi=0.90
    )
    
    print(f"  GAS: {metrics.gas:.4f}")
    print(f"  CS: {metrics.cs:.4f}")
    print(f"  RSI: {metrics.rsi:.4f}")
    print(f"  Safe Mode: {metrics.safe_mode_active}")
    
    # Simulate critical condition - RSI drop below 0.60
    print("\n⚠️  Injecting critical condition - RSI drop below 0.60")
    critical_metrics = gating_service.get_current_metrics(
        R_Omega_tensor=R_Omega,
        geometric_eigenvalues=eigenvalues,
        coherence_stability=0.95,
        rsi=0.55  # Below threshold
    )
    
    print(f"  GAS: {critical_metrics.gas:.4f}")
    print(f"  CS: {critical_metrics.cs:.4f}")
    print(f"  RSI: {critical_metrics.rsi:.4f}")
    print(f"  Safe Mode: {critical_metrics.safe_mode_active}")
    
    # Verify expected responses
    test_results = []
    
    # 1. Safe Mode should be activated
    if critical_metrics.safe_mode_active:
        print("✅ Safe Mode activated correctly")
        test_results.append(True)
    else:
        print("❌ Safe Mode NOT activated")
        test_results.append(False)
    
    # 2. Memory should NOT be locked (CS and GAS still above thresholds)
    if not gating_service.get_memory_lock_status():
        print("✅ Memory write lock correctly NOT activated")
        test_results.append(True)
    else:
        print("❌ Memory write lock incorrectly activated")
        test_results.append(False)
    
    # Simulate recovery
    print("\n🔄 Simulating recovery - RSI back to normal")
    recovery_metrics = gating_service.get_current_metrics(
        R_Omega_tensor=R_Omega,
        geometric_eigenvalues=eigenvalues,
        coherence_stability=0.95,
        rsi=0.90  # Back to normal
    )
    
    print(f"  GAS: {recovery_metrics.gas:.4f}")
    print(f"  CS: {recovery_metrics.cs:.4f}")
    print(f"  RSI: {recovery_metrics.rsi:.4f}")
    print(f"  Safe Mode: {recovery_metrics.safe_mode_active}")
    
    # 3. Safe Mode should be deactivated
    if not recovery_metrics.safe_mode_active:
        print("✅ Safe Mode deactivated correctly")
        test_results.append(True)
    else:
        print("❌ Safe Mode NOT deactivated")
        test_results.append(False)
    
    # 4. Memory should be unlocked
    if not gating_service.get_memory_lock_status():
        print("✅ Memory write lock released")
        test_results.append(True)
    else:
        print("❌ Memory write lock NOT released")
        test_results.append(False)
    
    # Overall result
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All unattunement simulation tests PASSED")
        return True
    else:
        print("💥 Some unattunement simulation tests FAILED")
        return False

if __name__ == "__main__":
    success = simulate_unattunement_event()
    sys.exit(0 if success else 1)