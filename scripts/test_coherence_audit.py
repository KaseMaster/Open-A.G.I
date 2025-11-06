#!/usr/bin/env python3
"""
Coherence consistency audit under random harmonic variance
"""

import sys
import os
import time
import random
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import make_snapshot, compute_coherence_score, recursive_validate

def add_noise_to_signal(values: np.ndarray, noise_level: float) -> np.ndarray:
    """Add random noise to a signal"""
    noise = np.random.normal(0, noise_level, len(values))
    return values + noise

def generate_harmonic_signal(freq: float, phase: float, duration: float = 0.5, sample_rate: int = 1024) -> tuple:
    """Generate a harmonic signal with optional noise"""
    t = np.linspace(0, duration, sample_rate)
    x = np.sin(2 * np.pi * freq * t + phase)
    return t, x

def run_coherence_consistency_audit():
    """Run coherence consistency audit under random harmonic variance"""
    print("🔍 Coherence Consistency Audit")
    print("=" * 40)
    
    # Test parameters
    num_tests = 50
    frequencies = [40.0, 50.0, 60.0, 100.0, 200.0]
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    
    print(f"🧪 Running {num_tests} tests with {len(frequencies)} frequencies and {len(noise_levels)} noise levels")
    
    # Results tracking
    results = {
        "consistent_coherence": 0,
        "inconsistent_coherence": 0,
        "high_coherence": 0,
        "low_coherence": 0,
        "total_calculations": 0
    }
    
    # Run tests
    for test_num in range(num_tests):
        # Randomly select parameters
        freq = random.choice(frequencies)
        noise_level = random.choice(noise_levels)
        phase_diff = random.uniform(0, 0.5)
        
        # Generate base signal
        t, base_signal = generate_harmonic_signal(freq, 0)
        
        # Generate related signals with phase difference and noise
        signal_a = add_noise_to_signal(base_signal, noise_level)
        signal_b = add_noise_to_signal(np.sin(2 * np.pi * freq * t + phase_diff), noise_level)
        signal_c = add_noise_to_signal(np.sin(2 * np.pi * freq * t + phase_diff * 2), noise_level)
        
        # Create snapshots
        snapshot_a = make_snapshot("node-A", t.tolist(), signal_a.tolist(), secret_key="keyA")
        snapshot_b = make_snapshot("node-B", t.tolist(), signal_b.tolist(), secret_key="keyB")
        snapshot_c = make_snapshot("node-C", t.tolist(), signal_c.tolist(), secret_key="keyC")
        
        # Calculate pairwise coherences
        cs_ab = compute_coherence_score(snapshot_a, [snapshot_b])
        cs_ac = compute_coherence_score(snapshot_a, [snapshot_c])
        cs_bc = compute_coherence_score(snapshot_b, [snapshot_c])
        
        # Check consistency (coherences should be similar for related signals)
        coherences = [cs_ab, cs_ac, cs_bc]
        mean_coherence = np.mean(coherences)
        std_coherence = np.std(coherences)
        
        # Consistency check: standard deviation should be low for related signals
        is_consistent = std_coherence < 0.1
        is_high_coherence = mean_coherence > 0.5
        
        # Update results
        results["total_calculations"] += 3
        if is_consistent:
            results["consistent_coherence"] += 1
        else:
            results["inconsistent_coherence"] += 1
            
        if is_high_coherence:
            results["high_coherence"] += 1
        else:
            results["low_coherence"] += 1
        
        # Print periodic updates
        if (test_num + 1) % 10 == 0:
            print(f"   Completed {test_num + 1}/{num_tests} tests")
    
    # Calculate final statistics
    consistent_rate = (results["consistent_coherence"] / num_tests) * 100
    high_coherence_rate = (results["high_coherence"] / (num_tests * 3)) * 100  # 3 coherences per test
    
    print(f"\n📊 Audit Results:")
    print(f"   Consistent Coherence: {results['consistent_coherence']}/{num_tests} ({consistent_rate:.1f}%)")
    print(f"   Inconsistent Coherence: {results['inconsistent_coherence']}/{num_tests} ({100-consistent_rate:.1f}%)")
    print(f"   High Coherence Signals: {results['high_coherence']}/{results['total_calculations']} ({high_coherence_rate:.1f}%)")
    print(f"   Low Coherence Signals: {results['low_coherence']}/{results['total_calculations']} ({100-high_coherence_rate:.1f}%)")
    
    # Test recursive validation with varying coherence
    print(f"\n🔍 Testing Recursive Validation...")
    validation_tests = 10
    valid_validations = 0
    
    for i in range(validation_tests):
        # Generate signals with varying coherence
        t = np.linspace(0, 0.5, 1024)
        base_freq = random.choice(frequencies)
        base_phase = random.uniform(0, 2 * np.pi)
        
        # Create snapshots with controlled coherence
        snapshots = []
        for j in range(3):
            phase_offset = j * 0.1  # Small phase differences for coherence
            noise = random.choice(noise_levels) * 0.1  # Low noise for coherence
            
            signal = np.sin(2 * np.pi * base_freq * t + base_phase + phase_offset)
            noisy_signal = add_noise_to_signal(signal, noise)
            
            snapshot = make_snapshot(f"node-{j}", t.tolist(), noisy_signal.tolist(), secret_key=f"key{j}")
            snapshots.append(snapshot)
        
        # Test with different thresholds
        threshold = random.uniform(0.1, 0.9)
        is_valid, proof_bundle = recursive_validate(snapshots, threshold=threshold)
        
        if is_valid:
            valid_validations += 1
    
    validation_success_rate = (valid_validations / validation_tests) * 100
    print(f"   Successful Validations: {valid_validations}/{validation_tests} ({validation_success_rate:.1f}%)")
    
    # Overall assessment
    print(f"\n🏆 Consistency Assessment:")
    if consistent_rate > 80:
        print("   ✅ Coherence calculation shows good consistency")
    else:
        print("   ⚠️  Coherence calculation shows some inconsistency")
        
    if high_coherence_rate > 30:
        print("   ✅ System can achieve high coherence with proper signals")
    else:
        print("   ⚠️  System struggles to achieve high coherence")
        
    if validation_success_rate > 50:
        print("   ✅ Recursive validation works correctly")
    else:
        print("   ⚠️  Recursive validation needs improvement")
    
    print("\n✅ Coherence consistency audit completed successfully!")

if __name__ == "__main__":
    run_coherence_consistency_audit()