#!/usr/bin/env python3
"""
Unit tests for harmonic validation module
"""

import sys
import os
import numpy as np

# Add parent directory to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import (
    compute_spectrum, 
    pairwise_coherence, 
    compute_coherence_score, 
    recursive_validate, 
    make_snapshot,
    HarmonicSnapshot
)


def test_compute_spectrum():
    """Test spectrum computation with a simple sine wave"""
    # Generate a test signal
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave
    
    spectrum = compute_spectrum(t, x)
    
    # Check that we get a spectrum back
    assert len(spectrum) > 0
    assert all(len(pair) == 2 for pair in spectrum)  # Each element is a (freq, amp) pair
    
    # Find the peak frequency (should be around 50 Hz)
    frequencies, amplitudes = zip(*spectrum)
    peak_idx = np.argmax(amplitudes)
    peak_freq = frequencies[peak_idx]
    
    # Peak should be close to 50 Hz (within 5 Hz tolerance)
    assert abs(peak_freq - 50) < 5
    
    print("✓ test_compute_spectrum passed")


def test_pairwise_coherence_identical():
    """Test coherence between identical signals"""
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 50 * t)
    y = np.sin(2 * np.pi * 50 * t)  # Identical signal
    
    coherence = pairwise_coherence(x, y, 1000)
    
    # Coherence should be higher for identical signals
    assert coherence > 0.15
    print("✓ test_pairwise_coherence_identical passed")


def test_pairwise_coherence_different():
    """Test coherence between different signals"""
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 50 * t)
    y = np.sin(2 * np.pi * 150 * t)  # Different frequency
    
    coherence = pairwise_coherence(x, y, 1000)
    
    # Coherence should be lower for different signals
    assert coherence < 0.5
    print("✓ test_pairwise_coherence_different passed")


def test_make_snapshot():
    """Test creating a harmonic snapshot"""
    t = np.linspace(0, 1, 1000).tolist()
    x = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000)).tolist()
    
    snapshot = make_snapshot("test-node", t, x, "secret-key")
    
    # Check that all required fields are present
    assert snapshot.node_id == "test-node"
    assert len(snapshot.times) == len(t)
    assert len(snapshot.values) == len(x)
    assert len(snapshot.spectrum) > 0
    assert len(snapshot.spectrum_hash) > 0
    assert snapshot.signature is not None
    assert "phi" in snapshot.phi_params
    assert "lambda" in snapshot.phi_params
    
    print("✓ test_make_snapshot passed")


def test_compute_coherence_score():
    """Test computing coherence score between snapshots"""
    # Create test snapshots
    t = np.linspace(0, 1, 1000)
    
    # Coherent signals
    x1 = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))
    x2 = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))
    
    # Different signal
    x3 = np.sin(2 * np.pi * 100 * t) + 0.2 * np.random.randn(len(t))
    
    snapshot1 = make_snapshot("node-1", t.tolist(), x1.tolist())
    snapshot2 = make_snapshot("node-2", t.tolist(), x2.tolist())
    snapshot3 = make_snapshot("node-3", t.tolist(), x3.tolist())
    
    # Test coherence between coherent snapshots
    coherence_coherent = pairwise_coherence(x1, x2, 1000)
    assert coherence_coherent > 0.15
    
    # Test coherence between different snapshots
    coherence_different = pairwise_coherence(x1, x3, 1000)
    assert coherence_different < 0.3
    
    print("✓ test_compute_coherence_score passed")


def test_recursive_validate():
    """Test recursive validation function"""
    t = np.linspace(0, 1, 1000)
    
    # Create coherent snapshots
    x1 = np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(len(t))
    x2 = np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(len(t))
    x3 = np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(len(t))
    
    snapshot1 = make_snapshot("node-1", t.tolist(), x1.tolist())
    snapshot2 = make_snapshot("node-2", t.tolist(), x2.tolist())
    snapshot3 = make_snapshot("node-3", t.tolist(), x3.tolist())
    
    bundle = [snapshot1, snapshot2, snapshot3]
    
    # Validate the bundle
    is_valid, proof_bundle = recursive_validate(bundle, threshold=0.15)
    
    # Should have some coherence due to similar signals
    assert proof_bundle is not None
    assert proof_bundle.aggregated_CS > 0.15
    assert is_valid == True
    
    print("✓ test_recursive_validate passed")


def run_all_tests():
    """Run all tests"""
    print("Running harmonic validation tests...")
    
    test_compute_spectrum()
    test_pairwise_coherence_identical()
    test_pairwise_coherence_different()
    test_make_snapshot()
    test_compute_coherence_score()
    test_recursive_validate()
    
    print("\nAll tests passed! ✅")


if __name__ == "__main__":
    run_all_tests()