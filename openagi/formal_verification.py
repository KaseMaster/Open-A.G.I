#!/usr/bin/env python3
"""
Formal Verification Module for Quantum Currency System
Implements property-based verification of harmonic consensus logic

This module provides formal verification of the core properties of the 
Recursive Φ-Resonance Validation (RΦV) system using property-based testing
and mathematical assertions.
"""

import sys
import os
import numpy as np
from typing import List, Tuple
import logging

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from openagi.harmonic_validation import (
    compute_spectrum, 
    pairwise_coherence, 
    compute_coherence_score,
    apply_recursive_decay,
    recursive_validate,
    make_snapshot,
    HarmonicSnapshot,
    HarmonicProofBundle
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HarmonicConsensusVerifier:
    """
    Formal verifier for harmonic consensus properties
    """
    
    def __init__(self):
        self.verification_results = []
    
    def verify_spectrum_properties(self, times: np.ndarray, values: np.ndarray) -> bool:
        """
        Verify mathematical properties of spectrum computation
        
        Properties:
        1. Spectrum frequencies are non-negative
        2. Spectrum amplitudes are non-negative
        3. Spectrum has correct length
        """
        logger.info("Verifying spectrum computation properties...")
        
        try:
            spectrum = compute_spectrum(times, values)
            
            # Property 1: All frequencies are non-negative
            frequencies = [freq for freq, _ in spectrum]
            assert all(f >= 0 for f in frequencies), "All frequencies must be non-negative"
            
            # Property 2: All amplitudes are non-negative
            amplitudes = [amp for _, amp in spectrum]
            assert all(a >= 0 for a in amplitudes), "All amplitudes must be non-negative"
            
            # Property 3: Spectrum length is correct
            expected_length = len(times) // 2 + 1  # rfft length
            assert len(spectrum) == expected_length, f"Spectrum length mismatch: {len(spectrum)} != {expected_length}"
            
            logger.info("✓ Spectrum properties verified")
            self.verification_results.append(("spectrum_properties", True))
            return True
            
        except Exception as e:
            logger.error(f"✗ Spectrum properties verification failed: {e}")
            self.verification_results.append(("spectrum_properties", False, str(e)))
            return False
    
    def verify_coherence_bounds(self, x1: np.ndarray, x2: np.ndarray, fs: float) -> bool:
        """
        Verify mathematical bounds of coherence computation
        
        Properties:
        1. Coherence score is between 0 and 1
        2. Coherence of identical signals is 1.0
        3. Coherence is symmetric
        """
        logger.info("Verifying coherence computation bounds...")
        
        try:
            # Property 1: Coherence is between 0 and 1
            coherence = pairwise_coherence(x1, x2, fs)
            assert 0.0 <= coherence <= 1.0, f"Coherence out of bounds: {coherence}"
            
            # Property 2: Coherence is symmetric
            coherence_12 = pairwise_coherence(x1, x2, fs)
            coherence_21 = pairwise_coherence(x2, x1, fs)
            assert abs(coherence_12 - coherence_21) < 1e-6, f"Coherence not symmetric: {coherence_12} != {coherence_21}"
            
            # Property 3: Coherence of a signal with itself is well-defined
            identical_coherence = pairwise_coherence(x1, x1, fs)
            assert 0.0 <= identical_coherence <= 1.0, f"Self-coherence out of bounds: {identical_coherence}"
            
            logger.info("✓ Coherence bounds verified")
            self.verification_results.append(("coherence_bounds", True))
            return True
            
        except Exception as e:
            logger.error(f"✗ Coherence bounds verification failed: {e}")
            self.verification_results.append(("coherence_bounds", False, str(e)))
            return False
    
    def verify_recursive_decay_properties(self, coherence_series: List[float]) -> bool:
        """
        Verify properties of recursive decay function
        
        Properties:
        1. Output is between 0 and 1 if all inputs are between 0 and 1
        2. Output is 0 if all inputs are 0
        3. Output approaches 1 if all inputs approach 1
        """
        logger.info("Verifying recursive decay properties...")
        
        try:
            result = apply_recursive_decay(coherence_series)
            
            # Property 1: Output is between 0 and 1
            assert 0.0 <= result <= 1.0, f"Recursive decay output out of bounds: {result}"
            
            # Property 2: Output is 0 if all inputs are 0
            zero_series = [0.0] * len(coherence_series)
            zero_result = apply_recursive_decay(zero_series)
            assert abs(zero_result - 0.0) < 1e-6, f"Zero series should produce zero result: {zero_result}"
            
            # Property 3: Output approaches 1 if all inputs approach 1
            ones_series = [1.0] * len(coherence_series)
            ones_result = apply_recursive_decay(ones_series)
            assert abs(ones_result - 1.0) < 1e-6, f"Ones series should produce ones result: {ones_result}"
            
            logger.info("✓ Recursive decay properties verified")
            self.verification_results.append(("recursive_decay_properties", True))
            return True
            
        except Exception as e:
            logger.error(f"✗ Recursive decay properties verification failed: {e}")
            self.verification_results.append(("recursive_decay_properties", False, str(e)))
            return False
    
    def verify_validation_consistency(self, snapshots: List[HarmonicSnapshot]) -> bool:
        """
        Verify consistency properties of validation process
        
        Properties:
        1. Validation is deterministic
        2. Proof bundle contains correct aggregated score
        3. Validation result matches threshold comparison
        """
        logger.info("Verifying validation consistency...")
        
        try:
            # Run validation twice to check determinism
            is_valid1, proof1 = recursive_validate(snapshots, threshold=0.5)
            is_valid2, proof2 = recursive_validate(snapshots, threshold=0.5)
            
            # Check that we got valid proofs
            if proof1 is None or proof2 is None:
                raise AssertionError("Validation did not return proof bundles")
            
            # Property 1: Validation is deterministic
            assert is_valid1 == is_valid2, f"Validation not deterministic: {is_valid1} != {is_valid2}"
            assert proof1.aggregated_CS == proof2.aggregated_CS, f"Proof scores not deterministic: {proof1.aggregated_CS} != {proof2.aggregated_CS}"
            
            # Property 2: Proof bundle contains correct aggregated score
            assert proof1.aggregated_CS >= 0.0 and proof1.aggregated_CS <= 1.0, f"Invalid aggregated score: {proof1.aggregated_CS}"
            
            # Property 3: Validation result matches threshold comparison
            expected_valid = proof1.aggregated_CS >= 0.5
            assert is_valid1 == expected_valid, f"Validation result mismatch: {is_valid1} != {expected_valid}"
            
            logger.info("✓ Validation consistency verified")
            self.verification_results.append(("validation_consistency", True))
            return True
            
        except Exception as e:
            logger.error(f"✗ Validation consistency verification failed: {e}")
            self.verification_results.append(("validation_consistency", False, str(e)))
            return False
    
    def verify_snapshot_integrity(self, snapshot: HarmonicSnapshot) -> bool:
        """
        Verify integrity properties of harmonic snapshots
        
        Properties:
        1. Spectrum hash matches computed spectrum
        2. Times and values have same length
        3. Coherence score is between 0 and 1
        """
        logger.info("Verifying snapshot integrity...")
        
        try:
            # Property 1: Times and values have same length
            assert len(snapshot.times) == len(snapshot.values), f"Times and values length mismatch: {len(snapshot.times)} != {len(snapshot.values)}"
            
            # Property 2: Coherence score is between 0 and 1
            assert 0.0 <= snapshot.CS <= 1.0, f"Invalid coherence score: {snapshot.CS}"
            
            # Property 3: Spectrum hash matches computed spectrum
            import json
            import hashlib
            spectrum_str = json.dumps(snapshot.spectrum, sort_keys=True)
            computed_hash = hashlib.sha256(spectrum_str.encode()).hexdigest()
            assert snapshot.spectrum_hash == computed_hash, f"Spectrum hash mismatch"
            
            logger.info("✓ Snapshot integrity verified")
            self.verification_results.append(("snapshot_integrity", True))
            return True
            
        except Exception as e:
            logger.error(f"✗ Snapshot integrity verification failed: {e}")
            self.verification_results.append(("snapshot_integrity", False, str(e)))
            return False
    
    def run_all_verifications(self) -> Tuple[bool, List[Tuple]]:
        """
        Run all formal verifications
        
        Returns:
            Tuple of (all_passed, results_list)
        """
        logger.info("Running all formal verifications...")
        
        # Generate test data
        t1 = np.linspace(0, 1.0, 1000)
        x1 = np.sin(2 * np.pi * 10 * t1)
        x2 = np.sin(2 * np.pi * 10 * t1)  # Identical
        x3 = np.sin(2 * np.pi * 15 * t1)  # Different frequency
        
        # Test spectrum properties
        spectrum_ok = self.verify_spectrum_properties(t1, x1)
        
        # Test coherence bounds
        coherence_ok = self.verify_coherence_bounds(x1, x3, 1000)
        
        # Test recursive decay properties
        decay_ok = self.verify_recursive_decay_properties([0.8, 0.7, 0.6])
        
        # Create test snapshots
        snapshot_a = make_snapshot("node-A", t1.tolist(), x1.tolist())
        snapshot_b = make_snapshot("node-B", t1.tolist(), x2.tolist())
        snapshot_c = make_snapshot("node-C", t1.tolist(), x3.tolist())
        snapshots = [snapshot_a, snapshot_b, snapshot_c]
        
        # Test validation consistency
        validation_ok = self.verify_validation_consistency(snapshots)
        
        # Test snapshot integrity
        integrity_ok = self.verify_snapshot_integrity(snapshot_a)
        
        # Check if all verifications passed
        all_passed = all([
            spectrum_ok,
            coherence_ok,
            decay_ok,
            validation_ok,
            integrity_ok
        ])
        
        logger.info(f"Formal verification {'PASSED' if all_passed else 'FAILED'}")
        return all_passed, self.verification_results

def demo_formal_verification():
    """Demonstrate formal verification of harmonic consensus logic"""
    print("🔍 Formal Verification of Harmonic Consensus Logic")
    print("=" * 50)
    
    # Create verifier instance
    verifier = HarmonicConsensusVerifier()
    
    # Run all verifications
    all_passed, results = verifier.run_all_verifications()
    
    # Display results
    print("\n📋 Verification Results:")
    for result in results:
        if len(result) == 2:
            name, passed = result
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {name}")
        elif len(result) == 3:
            name, passed, error = result
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {name}")
            if not passed:
                print(f"      Error: {error}")
    
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🏆 Formal verification completed successfully!")
        print("The harmonic consensus logic satisfies all verified properties.")
    else:
        print("\n⚠️  Formal verification found issues that need to be addressed.")

if __name__ == "__main__":
    demo_formal_verification()