#!/usr/bin/env python3
"""
Demo script showing AFIP (Absolute Field Integrity Protocol) v1.0 in action
"""

import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.afip.orchestrator import AFIPOrchestrator

def run_afip_demo():
    """Run a demonstration of AFIP in action"""
    print("⚛️ AFIP v1.0 Demonstration")
    print("=" * 50)
    print("This demo will show AFIP executing the complete protocol...")
    print()
    time.sleep(2)
    
    # Configure AFIP for demo (faster execution)
    config = {
        "shard_count": 3,
        "tee_enabled": True,
        "prediction_cycles": 5,  # Reduced for demo
        "observation_period_days": 1  # Reduced for demo
    }
    
    # Initialize AFIP orchestrator
    print("🚀 Initializing AFIP v1.0 Orchestrator...")
    afip = AFIPOrchestrator(config)
    time.sleep(1)
    
    # Define demo nodes
    print("📡 Configuring QECS nodes...")
    nodes = [
        {"node_id": "node_alpha", "coherence_score": 0.98, 
         "qra_params": {"n": 1, "l": 0, "m": 0, "s": 0.5}},
        {"node_id": "node_beta", "coherence_score": 0.96,
         "qra_params": {"n": 2, "l": 1, "m": -1, "s": 0.7}},
        {"node_id": "node_gamma", "coherence_score": 0.97,
         "qra_params": {"n": 1, "l": 1, "m": 0, "s": 0.3}},
        {"node_id": "node_delta", "coherence_score": 0.95,
         "qra_params": {"n": 3, "l": 2, "m": 1, "s": 0.9}},
        {"node_id": "node_epsilon", "coherence_score": 0.99,
         "qra_params": {"n": 2, "l": 0, "m": 0, "s": 0.2}},
    ]
    time.sleep(1)
    
    # Define demo telemetry data
    print("📊 Loading telemetry data...")
    telemetry_data = [
        {"node_id": "node_alpha", "g_vector_magnitude": 0.5, "coherence": 0.98, "rsi": 0.92, "delta_h": 0.001},
        {"node_id": "node_alpha", "g_vector_magnitude": 0.7, "coherence": 0.97, "rsi": 0.91, "delta_h": 0.002},
        {"node_id": "node_alpha", "g_vector_magnitude": 0.9, "coherence": 0.96, "rsi": 0.93, "delta_h": 0.001},
        {"node_id": "node_beta", "g_vector_magnitude": 0.4, "coherence": 0.96, "rsi": 0.94, "delta_h": 0.0005},
        {"node_id": "node_beta", "g_vector_magnitude": 0.6, "coherence": 0.95, "rsi": 0.92, "delta_h": 0.001},
        {"node_id": "node_gamma", "g_vector_magnitude": 0.3, "coherence": 0.97, "rsi": 0.95, "delta_h": 0.0002},
        {"node_id": "node_delta", "g_vector_magnitude": 0.2, "coherence": 0.98, "rsi": 0.96, "delta_h": 0.0001},
        {"node_id": "node_epsilon", "g_vector_magnitude": 0.8, "coherence": 0.99, "rsi": 0.97, "delta_h": 0.0003},
    ] * 5  # Repeat for more data
    time.sleep(1)
    
    # First, try to auto-tune Phase II to fix the failures
    print("⚡ Auto-tuning Phase II parameters...")
    tuning_result = afip.auto_tune_phase_ii(telemetry_data, max_iterations=20)
    
    if tuning_result["success"]:
        print("✅ Phase II tuning successful!")
    else:
        print("⚠️ Phase II tuning did not fully converge, but will proceed with best parameters...")
        # Check if protocol proposal was triggered for self-evolution
        if tuning_result.get("protocol_proposal_triggered"):
            print("🔄 Protocol amendment proposal triggered for HSMF self-evolution")
    
    # Execute full AFIP protocol
    print("⚡ Executing Complete AFIP v1.0 Protocol...")
    print("   This may take a few moments...")
    print()
    
    start_time = time.time()
    final_report = afip.execute_full_afip_protocol(nodes, telemetry_data)
    execution_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 50)
    print("AFIP DEMO RESULTS")
    print("=" * 50)
    print(f"Overall Success: {'✅ PASS' if final_report['overall_success'] else '❌ FAIL'}")
    print(f"Final Status Code: {final_report['final_status_code']}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print()
    
    # Phase-by-phase results
    print("Phase Results:")
    print(f"  Phase I (Hardening): {'✅ PASS' if final_report['phase_i_success'] else '❌ FAIL'}")
    print(f"  Phase II (Predictive): {'✅ PASS' if final_report['phase_ii_success'] else '❌ FAIL'}")
    print(f"  CI/CD Validation: {'✅ PASS' if final_report['ci_cd_success'] else '❌ FAIL'}")
    print(f"  Phase III (Evolution): {'✅ PASS' if final_report['phase_iii_success'] else '❌ FAIL'}")
    print()
    
    # Final assessment
    if final_report['overall_success']:
        print("🎉 CONGRATULATIONS! 🎉")
        print("   QECS has successfully passed all AFIP v1.0 protocols")
        print("   and is now production-ready with:")
        print("   ✅ Zero-dissonance deployment validated")
        print("   ✅ Predictive governance mechanisms active")
        print("   ✅ Autonomous evolution protocols enabled")
        print("   ✅ Final coherence lock achieved")
        print()
        print("   Your quantum economic system is ready for autonomous operation!")
    else:
        print("⚠️ QECS requires additional tuning before production deployment")
        print("   Review the AFIP execution report for details")
    
    return 0 if final_report['overall_success'] else 1

if __name__ == "__main__":
    sys.exit(run_afip_demo())