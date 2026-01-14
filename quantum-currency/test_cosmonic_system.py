#!/usr/bin/env python3
"""
Test script for Cosmonic Verification System
"""

import sys
import os

# Add the current directory to the path
sys.path.append('.')

def main():
    print("🌌 Testing Cosmonic Verification System")
    
    # Import the cosmonic verification system
    try:
        from src.core.cosmonic_verification import CosmonicVerificationSystem
        print("✅ Cosmonic Verification System imported successfully")
        
        # Create cosmonic verification system
        cosmonic_system = CosmonicVerificationSystem()
        print("✅ Cosmonic Verification System instantiated successfully")
        
        # Run cosmoverification metrics
        print("\n📊 Running Cosmoverification Metrics...")
        metrics_result = cosmonic_system.cosmoverification_metrics()
        print(f"   Status: {metrics_result['status']}")
        metrics = metrics_result['metrics']
        print(f"   H_internal: {metrics['H_internal']:.4f}")
        print(f"   CAF: {metrics['CAF']:.4f}")
        print(f"   Entropy Rate: {metrics['entropy_rate']:.4f}")
        
        # Run full system verification
        print("\n🔍 Running Full System Verification...")
        verification_result = cosmonic_system.full_system_verification()
        print(f"   Status: {verification_result.status}")
        print(f"   Issues: {len(verification_result.issues)}")
        print(f"   Recommendations: {len(verification_result.recommendations)}")
        
        # Run autonomous coherence optimization
        print("\n⚡ Running Autonomous Coherence Optimization...")
        optimization_result = cosmonic_system.autonomous_coherence_optimization()
        print(f"   Status: {optimization_result['status']}")
        print(f"   Optimizations: {len(optimization_result['optimizations'])}")
        
        # Generate cosmic coherence report
        print("\n📋 Generating Cosmic Coherence Report...")
        report = cosmonic_system.generate_cosmic_coherence_report()
        print(f"   Phase: {report['phase']}")
        print(f"   H_internal: {report['H_internal']:.4f}")
        print(f"   CAF: {report['CAF']:.4f}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()