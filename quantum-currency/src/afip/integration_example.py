#!/usr/bin/env python3
"""
Integration example showing how AFIP works with the existing IACE v2.0 orchestrator
"""

import sys
import os
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing QECS modules
try:
    from iace_v2_orchestrator import QECSOrchestrator
    from src.monitoring.telemetry_streamer import telemetry_streamer
    QECS_AVAILABLE = True
except ImportError:
    QECS_AVAILABLE = False
    print("QECS modules not available, using mock implementations")

# Import AFIP
from afip.orchestrator import AFIPOrchestrator

def integrate_afip_with_iace():
    """Demonstrate integration of AFIP with IACE v2.0"""
    print("⚛️ Integrating AFIP with IACE v2.0")
    print("=" * 40)
    
    # First, run IACE v2.0 to get system state
    if QECS_AVAILABLE:
        print("🚀 Running IACE v2.0 orchestration...")
        iace = QECSOrchestrator()
        
        # Run IACE phases
        iace.phase_i_core_system()
        iace.phase_ii_iii_transaction_security()
        
        # Get system state from IACE
        system_state = {
            "final_status": "200_COHERENT_LOCK" if all(s == 0 for s in iace.phase_status.values()) else "500_CRITICAL_DISSONANCE",
            "delta_lambda": 0.001,  # Simulated value
            "C_system": 0.98,       # Simulated value
            "GAS_target": 0.95      # Simulated value
        }
        
        # Get telemetry data
        telemetry_data = iace.kpi_history
    else:
        # Mock system state for demonstration
        system_state = {
            "final_status": "200_COHERENT_LOCK",
            "delta_lambda": 0.001,
            "C_system": 0.98,
            "GAS_target": 0.95
        }
        
        # Mock telemetry data
        telemetry_data = [
            {"node_id": "node_001", "g_vector_magnitude": 0.5, "coherence": 0.98, "rsi": 0.92, "delta_h": 0.001},
            {"node_id": "node_001", "g_vector_magnitude": 0.7, "coherence": 0.97, "rsi": 0.91, "delta_h": 0.002},
            {"node_id": "node_001", "g_vector_magnitude": 0.9, "coherence": 0.96, "rsi": 0.93, "delta_h": 0.001},
            {"node_id": "node_002", "g_vector_magnitude": 0.4, "coherence": 0.96, "rsi": 0.94, "delta_h": 0.0005},
            {"node_id": "node_002", "g_vector_magnitude": 0.6, "coherence": 0.95, "rsi": 0.92, "delta_h": 0.001},
            {"node_id": "node_003", "g_vector_magnitude": 0.3, "coherence": 0.97, "rsi": 0.95, "delta_h": 0.0002},
        ] * 10
    
    print(f"✅ IACE v2.0 completed with status: {system_state['final_status']}")
    
    # Now run AFIP using the system state from IACE
    print("\n⚛️ Running AFIP v1.0 Protocol...")
    
    # Configure AFIP
    afip_config = {
        "shard_count": 3,
        "tee_enabled": True,
        "prediction_cycles": 10,
        "observation_period_days": 7
    }
    
    # Initialize AFIP orchestrator
    afip = AFIPOrchestrator(afip_config)
    
    # Define nodes (in a real system, these would come from the network discovery)
    nodes = [
        {"node_id": "node_001", "coherence_score": 0.98, 
         "qra_params": {"n": 1, "l": 0, "m": 0, "s": 0.5}},
        {"node_id": "node_002", "coherence_score": 0.96,
         "qra_params": {"n": 2, "l": 1, "m": -1, "s": 0.7}},
        {"node_id": "node_003", "coherence_score": 0.97,
         "qra_params": {"n": 1, "l": 1, "m": 0, "s": 0.3}},
        {"node_id": "node_004", "coherence_score": 0.95,
         "qra_params": {"n": 3, "l": 2, "m": 1, "s": 0.9}},
    ]
    
    # Execute full AFIP protocol
    final_report = afip.execute_full_afip_protocol(nodes, telemetry_data)
    
    # Print integration results
    print("\n" + "=" * 50)
    print("IACE + AFIP INTEGRATION RESULTS")
    print("=" * 50)
    print(f"IACE Final Status: {system_state['final_status']}")
    print(f"AFIP Overall Success: {'✅' if final_report['overall_success'] else '❌'}")
    print(f"AFIP Final Status Code: {final_report['final_status_code']}")
    print(f"QECS Production Ready: {'✅ YES' if final_report['overall_success'] else '❌ NO'}")
    
    if final_report['overall_success']:
        print("\n🎉 QECS IS NOW PRODUCTION READY! 🎉")
        print("   - Zero-dissonance deployment validated")
        print("   - Predictive governance mechanisms active")
        print("   - Autonomous evolution protocols enabled")
        print("   - Final coherence lock achieved")
    else:
        print("\n⚠️ QECS REQUIRES ADDITIONAL TUNING ⚠️")
        print("   - Review AFIP execution report for details")
    
    return final_report

def main():
    """Main function to run the integration example"""
    try:
        result = integrate_afip_with_iace()
        return 0 if result.get('overall_success', False) else 1
    except Exception as e:
        print(f"Error during integration: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())