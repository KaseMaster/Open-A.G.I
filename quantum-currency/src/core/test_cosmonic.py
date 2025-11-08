#!/usr/bin/env python3
"""
Test script for Cosmonic Verification System
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.cosmonic_verification import CosmonicVerificationSystem

def main():
    print("🌌 Testing Cosmonic Verification System")
    
    # Create cosmonic verification system
    cosmonic_system = CosmonicVerificationSystem()
    
    # Run cosmoverification metrics
    print("\n📊 Running Cosmoverification Metrics...")
    metrics_result = cosmonic_system.cosmoverification_metrics()
    print(f"   Status: {metrics_result['status']}")
    metrics = metrics_result['metrics']
    print(f"   H_internal: {metrics['H_internal']:.4f}")
    print(f"   CAF: {metrics['CAF']:.4f}")
    print(f"   Entropy Rate: {metrics['entropy_rate']:.4f}")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()