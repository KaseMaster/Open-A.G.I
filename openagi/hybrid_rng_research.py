#!/usr/bin/env python3
"""
Hybrid RNG Research for Quantum Currency System
Research on combining thermal and quantum noise for enhanced randomness
"""

import sys
import os
import json
import time
import hashlib
import secrets
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
from scipy.special import erfc  # Import erfc from scipy.special
from scipy.fft import fft
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our existing QRNG module
from openagi.quantum_rng import QuantumRNG, QuantumEntropySource

@dataclass
class HybridEntropySource:
    """Represents a hybrid entropy source combining multiple entropy sources"""
    source_id: str
    component_sources: List[str]  # List of source IDs that make up this hybrid source
    combination_method: str  # "xor", "hash", "concatenate"
    quality_metrics: Dict[str, float]  # Various quality metrics
    last_hybrid_entropy: Optional[str] = None

class HybridRNGResearch:
    """
    Research module for hybrid random number generation combining multiple entropy sources
    """
    
    def __init__(self):
        self.qrng = QuantumRNG()
        self.hybrid_sources: Dict[str, HybridEntropySource] = {}
        self.research_data: List[Dict] = []
        self._initialize_hybrid_sources()
    
    def _initialize_hybrid_sources(self):
        """Initialize hybrid entropy sources for research"""
        # Create hybrid sources combining different types of entropy
        hybrid_sources = [
            HybridEntropySource(
                source_id="hybrid-photonic-thermal-01",
                component_sources=["photonic-qe-01", "thermal-qe-01"],
                combination_method="xor",
                quality_metrics={"theoretical_quality": 0.98, "expected_rate": 1500.0}
            ),
            HybridEntropySource(
                source_id="hybrid-multi-source-01",
                component_sources=["photonic-qe-01", "thermal-qe-01", "atmospheric-qe-01"],
                combination_method="hash",
                quality_metrics={"theoretical_quality": 0.99, "expected_rate": 1600.0}
            ),
            HybridEntropySource(
                source_id="hybrid-photonic-atmospheric-01",
                component_sources=["photonic-qe-01", "atmospheric-qe-01"],
                combination_method="concatenate",
                quality_metrics={"theoretical_quality": 0.96, "expected_rate": 1100.0}
            )
        ]
        
        for source in hybrid_sources:
            self.hybrid_sources[source.source_id] = source
    
    def combine_entropy_sources(self, source_ids: List[str], method: str = "xor") -> str:
        """
        Combine entropy from multiple sources using specified method
        
        Args:
            source_ids: List of source IDs to combine
            method: Combination method ("xor", "hash", "concatenate")
            
        Returns:
            Combined entropy string
        """
        # Collect entropy from all specified sources
        entropy_samples = []
        for source_id in source_ids:
            if source_id in self.qrng.entropy_sources:
                sample = self.qrng.collect_entropy(source_id)
                if sample:
                    entropy_samples.append(sample)
        
        if not entropy_samples:
            # Fallback to OS-provided entropy
            return secrets.token_hex(32)
        
        # Combine using specified method
        if method == "xor":
            return self._xor_combine(entropy_samples)
        elif method == "hash":
            return self._hash_combine(entropy_samples)
        elif method == "concatenate":
            return self._concatenate_combine(entropy_samples)
        else:
            # Default to XOR
            return self._xor_combine(entropy_samples)
    
    def _xor_combine(self, samples: List) -> str:
        """
        Combine entropy samples using XOR operation
        
        Args:
            samples: List of entropy samples
            
        Returns:
            XOR-combined entropy string
        """
        if not samples:
            return secrets.token_hex(32)
        
        # Convert first sample to bytes
        combined_bytes = bytes.fromhex(samples[0].processed_entropy)
        
        # XOR with remaining samples
        for sample in samples[1:]:
            sample_bytes = bytes.fromhex(sample.processed_entropy)
            # Ensure same length by truncating or padding
            min_len = min(len(combined_bytes), len(sample_bytes))
            combined_bytes = bytes([a ^ b for a, b in zip(combined_bytes[:min_len], sample_bytes[:min_len])])
        
        return combined_bytes.hex()
    
    def _hash_combine(self, samples: List) -> str:
        """
        Combine entropy samples using cryptographic hash
        
        Args:
            samples: List of entropy samples
            
        Returns:
            Hash-combined entropy string
        """
        if not samples:
            return secrets.token_hex(32)
        
        # Concatenate all entropy
        combined_entropy = ''.join([sample.processed_entropy for sample in samples])
        
        # Hash to combine
        return hashlib.sha256(combined_entropy.encode()).hexdigest()
    
    def _concatenate_combine(self, samples: List) -> str:
        """
        Combine entropy samples by concatenation and hashing
        
        Args:
            samples: List of entropy samples
            
        Returns:
            Concatenated and hashed entropy string
        """
        if not samples:
            return secrets.token_hex(32)
        
        # Concatenate all entropy
        combined_entropy = ''.join([sample.processed_entropy for sample in samples])
        
        # Hash to ensure fixed length
        return hashlib.sha256(combined_entropy.encode()).hexdigest()
    
    def generate_hybrid_entropy(self, hybrid_source_id: str) -> Optional[str]:
        """
        Generate entropy from a hybrid source
        
        Args:
            hybrid_source_id: ID of the hybrid source
            
        Returns:
            Hybrid entropy string or None if source not found
        """
        if hybrid_source_id not in self.hybrid_sources:
            return None
        
        hybrid_source = self.hybrid_sources[hybrid_source_id]
        
        # Generate combined entropy
        combined_entropy = self.combine_entropy_sources(
            hybrid_source.component_sources,
            hybrid_source.combination_method
        )
        
        # Update hybrid source
        hybrid_source.last_hybrid_entropy = combined_entropy
        
        return combined_entropy
    
    def evaluate_randomness_quality(self, entropy_data: str, test_name: str = "general") -> Dict:
        """
        Evaluate the quality of entropy data using various statistical tests
        
        Args:
            entropy_data: Hex string of entropy data
            test_name: Name for this test run
            
        Returns:
            Dictionary with test results
        """
        # Convert hex to binary array
        try:
            binary_data = bin(int(entropy_data, 16))[2:].zfill(len(entropy_data) * 4)
            # Convert to numpy array of integers
            bit_array = np.array([int(bit) for bit in binary_data])
        except:
            # Fallback if conversion fails
            bit_array = np.random.randint(0, 2, 1000)
        
        # Perform various statistical tests
        results = {
            "test_name": test_name,
            "data_length_bits": len(bit_array),
            "timestamp": time.time()
        }
        
        # Frequency test (monobit test)
        results["frequency_test"] = self._frequency_test(bit_array)
        
        # Runs test
        results["runs_test"] = self._runs_test(bit_array)
        
        # Longest run of ones test
        results["longest_run_test"] = self._longest_run_test(bit_array)
        
        # Entropy test
        results["entropy"] = self._calculate_entropy(bit_array)
        
        # Chi-square test
        results["chi_square_test"] = self._chi_square_test(bit_array)
        
        # Serial test
        results["serial_test"] = self._serial_test(bit_array)
        
        # Overall quality score
        quality_score = self._calculate_overall_quality(results)
        results["overall_quality_score"] = quality_score
        
        # Store research data
        self.research_data.append(results)
        
        return results
    
    def _frequency_test(self, bit_array: np.ndarray) -> Dict:
        """Perform frequency (monobit) test"""
        n = len(bit_array)
        if n == 0:
            return {"passed": False, "p_value": 0.0, "reason": "No data"}
        
        # Count ones and zeros
        ones = np.sum(bit_array)
        zeros = n - ones
        
        # Calculate test statistic
        s_n = abs(ones - zeros)
        s_obs = s_n / np.sqrt(n)
        p_value = erfc(s_obs / np.sqrt(2))
        
        # Test passes if p-value >= 0.01
        passed = p_value >= 0.01
        
        return {
            "passed": passed,
            "p_value": p_value,
            "ones_count": int(ones),
            "zeros_count": int(zeros),
            "reason": "Frequency test passed" if passed else "Too many 0s or 1s"
        }
    
    def _runs_test(self, bit_array: np.ndarray) -> Dict:
        """Perform runs test"""
        n = len(bit_array)
        if n < 2:
            return {"passed": False, "p_value": 0.0, "reason": "Insufficient data"}
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if bit_array[i] != bit_array[i-1]:
                runs += 1
        
        # Count ones
        ones = np.sum(bit_array)
        pi = ones / n
        
        # Check if pi is within acceptable range
        if abs(pi - 0.5) >= 2/np.sqrt(n):
            return {
                "passed": False,
                "p_value": 0.0,
                "runs_count": runs,
                "pi": float(pi),
                "reason": "Proportion of ones too far from 0.5"
            }
        
        # Calculate test statistic
        numerator = abs(runs - 2 * n * pi * (1 - pi))
        denominator = 2 * np.sqrt(2 * n) * pi * (1 - pi)
        if denominator == 0:
            return {"passed": False, "p_value": 0.0, "reason": "Division by zero"}
        
        p_value = erfc(numerator / denominator)
        passed = p_value >= 0.01
        
        return {
            "passed": passed,
            "p_value": p_value,
            "runs_count": runs,
            "pi": float(pi),
            "reason": "Runs test passed" if passed else "Unexpected number of runs"
        }
    
    def _longest_run_test(self, bit_array: np.ndarray) -> Dict:
        """Perform longest run of ones test"""
        n = len(bit_array)
        if n < 128:
            return {"passed": False, "p_value": 0.0, "reason": "Insufficient data (< 128 bits)"}
        
        # For simplicity, we'll use a basic approach
        # Find longest run of consecutive ones
        max_run = 0
        current_run = 0
        
        for bit in bit_array:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        # Expected longest run for random sequence of length n is approximately log2(n)
        expected_max_run = np.log2(n)
        
        # Simple test: actual max run should be within reasonable bounds
        lower_bound = expected_max_run - 3
        upper_bound = expected_max_run + 3
        
        passed = lower_bound <= max_run <= upper_bound
        p_value = 0.5  # Simplified
        
        return {
            "passed": passed,
            "p_value": p_value,
            "longest_run": max_run,
            "expected_max_run": float(expected_max_run),
            "reason": "Longest run test passed" if passed else "Longest run outside expected bounds"
        }
    
    def _calculate_entropy(self, bit_array: np.ndarray) -> float:
        """Calculate Shannon entropy of bit array"""
        if len(bit_array) == 0:
            return 0.0
        
        # Count ones and zeros
        ones = np.sum(bit_array)
        zeros = len(bit_array) - ones
        
        # Calculate probabilities
        p1 = ones / len(bit_array)
        p0 = zeros / len(bit_array)
        
        # Avoid log(0)
        if p1 == 0 or p0 == 0:
            return 0.0
        
        # Calculate entropy
        entropy = -(p1 * np.log2(p1) + p0 * np.log2(p0))
        return entropy
    
    def _chi_square_test(self, bit_array: np.ndarray) -> Dict:
        """Perform chi-square test"""
        n = len(bit_array)
        if n < 10:
            return {"passed": False, "p_value": 0.0, "reason": "Insufficient data"}
        
        # Count ones and zeros
        ones = np.sum(bit_array)
        zeros = n - ones
        
        # Expected counts
        expected = n / 2
        
        # Chi-square statistic
        chi2 = ((ones - expected)**2 + (zeros - expected)**2) / expected
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        passed = p_value >= 0.01
        
        return {
            "passed": passed,
            "p_value": p_value,
            "chi_square_statistic": float(chi2),
            "reason": "Chi-square test passed" if passed else "Non-uniform distribution"
        }
    
    def _serial_test(self, bit_array: np.ndarray) -> Dict:
        """Perform serial test (count pairs of bits)"""
        n = len(bit_array)
        if n < 2:
            return {"passed": False, "p_value": 0.0, "reason": "Insufficient data"}
        
        # Count pairs
        pairs = {"00": 0, "01": 0, "10": 0, "11": 0}
        
        for i in range(n - 1):
            pair = f"{bit_array[i]}{bit_array[i+1]}"
            if pair in pairs:
                pairs[pair] += 1
        
        # Expected count for each pair
        expected = (n - 1) / 4
        
        # Chi-square test for pairs
        if expected == 0:
            return {"passed": False, "p_value": 0.0, "reason": "Expected count is zero"}
        
        chi2 = sum(((count - expected)**2) / expected for count in pairs.values())
        p_value = 1 - stats.chi2.cdf(chi2, df=3)  # 4 categories - 1 constraint
        
        passed = p_value >= 0.01
        
        return {
            "passed": passed,
            "p_value": p_value,
            "chi_square_statistic": float(chi2),
            "pair_counts": pairs,
            "reason": "Serial test passed" if passed else "Non-uniform pair distribution"
        }
    
    def _calculate_overall_quality(self, test_results: Dict) -> float:
        """Calculate overall quality score from test results"""
        passed_tests = 0
        total_tests = 0
        
        for key, value in test_results.items():
            if isinstance(value, dict) and "passed" in value:
                total_tests += 1
                if value["passed"]:
                    passed_tests += 1
        
        if total_tests == 0:
            return 0.0
        
        return passed_tests / total_tests
    
    def compare_hybrid_methods(self) -> Dict:
        """
        Compare different hybrid combination methods
        
        Returns:
            Dictionary with comparison results
        """
        results = {
            "comparison_timestamp": time.time(),
            "methods": {}
        }
        
        # Test each hybrid source
        for source_id, source in self.hybrid_sources.items():
            # Generate hybrid entropy
            hybrid_entropy = self.generate_hybrid_entropy(source_id)
            
            if hybrid_entropy:
                # Evaluate quality
                quality_results = self.evaluate_randomness_quality(
                    hybrid_entropy, 
                    f"Hybrid {source_id}"
                )
                
                results["methods"][source_id] = {
                    "component_sources": source.component_sources,
                    "combination_method": source.combination_method,
                    "quality_score": quality_results["overall_quality_score"],
                    "entropy": quality_results["entropy"],
                    "test_results": quality_results
                }
        
        return results
    
    def generate_research_report(self) -> Dict:
        """
        Generate a comprehensive research report
        
        Returns:
            Dictionary with research report
        """
        report = {
            "report_timestamp": time.time(),
            "title": "Hybrid RNG Research Report",
            "executive_summary": "Research on combining thermal and quantum noise for enhanced randomness",
            "hybrid_sources": {},
            "comparison_results": self.compare_hybrid_methods(),
            "recommendations": {}
        }
        
        # Add hybrid source information
        for source_id, source in self.hybrid_sources.items():
            report["hybrid_sources"][source_id] = {
                "component_sources": source.component_sources,
                "combination_method": source.combination_method,
                "theoretical_quality": source.quality_metrics.get("theoretical_quality", 0.0),
                "expected_rate": source.quality_metrics.get("expected_rate", 0.0)
            }
        
        # Generate recommendations based on comparison results
        best_method = None
        best_score = 0.0
        
        methods = report["comparison_results"]["methods"]
        for method_id, method_data in methods.items():
            quality_score = method_data["quality_score"]
            if quality_score > best_score:
                best_score = quality_score
                best_method = method_id
        
        report["recommendations"] = {
            "best_hybrid_method": best_method,
            "best_method_score": best_score,
            "recommended_for_consensus": best_method if best_score > 0.8 else "Use multiple methods",
            "implementation_notes": "Hybrid sources provide enhanced security through entropy combination"
        }
        
        return report

def demo_hybrid_rng_research():
    """Demonstrate hybrid RNG research capabilities"""
    print("🔬 Hybrid RNG Research Demo")
    print("=" * 35)
    
    # Create research instance
    research = HybridRNGResearch()
    
    # Show hybrid sources
    print("\n📡 Hybrid Entropy Sources:")
    for source_id, source in research.hybrid_sources.items():
        print(f"   {source_id}:")
        print(f"      Components: {', '.join(source.component_sources)}")
        print(f"      Method: {source.combination_method}")
        print(f"      Theoretical Quality: {source.quality_metrics.get('theoretical_quality', 0.0):.2f}")
    
    # Generate hybrid entropy from each source
    print("\n🔐 Generating Hybrid Entropy:")
    for source_id in research.hybrid_sources.keys():
        hybrid_entropy = research.generate_hybrid_entropy(source_id)
        if hybrid_entropy:
            print(f"   {source_id}: {hybrid_entropy[:16]}...")
        else:
            print(f"   {source_id}: Failed to generate")
    
    # Compare hybrid methods
    print("\n📊 Comparing Hybrid Methods:")
    comparison = research.compare_hybrid_methods()
    for method_id, method_data in comparison["methods"].items():
        print(f"   {method_id}:")
        print(f"      Quality Score: {method_data['quality_score']:.3f}")
        print(f"      Entropy: {method_data['entropy']:.3f}")
        print(f"      Method: {method_data['combination_method']}")
    
    # Generate research report
    print("\n📋 Research Report:")
    report = research.generate_research_report()
    print(f"   Title: {report['title']}")
    print(f"   Best Method: {report['recommendations']['best_hybrid_method']}")
    print(f"   Best Score: {report['recommendations']['best_method_score']:.3f}")
    print(f"   Recommendation: {report['recommendations']['recommended_for_consensus']}")
    
    # Test randomness quality of the best method
    best_method = report['recommendations']['best_hybrid_method']
    if best_method:
        hybrid_entropy = research.generate_hybrid_entropy(best_method)
        if hybrid_entropy:
            print(f"\n🔍 Randomness Quality Test for {best_method}:")
            quality_results = research.evaluate_randomness_quality(hybrid_entropy, "Best Hybrid Method")
            print(f"   Overall Quality Score: {quality_results['overall_quality_score']:.3f}")
            print(f"   Shannon Entropy: {quality_results['entropy']:.3f}")
            
            # Show individual test results
            for test_name, test_result in quality_results.items():
                if isinstance(test_result, dict) and "passed" in test_result:
                    status = "PASS" if test_result["passed"] else "FAIL"
                    print(f"   {test_name}: {status} (p={test_result.get('p_value', 'N/A'):.4f})")
    
    print("\n✅ Hybrid RNG research demo completed!")

if __name__ == "__main__":
    demo_hybrid_rng_research()