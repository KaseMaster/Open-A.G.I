#!/usr/bin/env python3
"""
Runner script for IACE v2.0 QECS Orchestration Engine
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("⚛️ IACE v2.0 QECS Orchestration Engine")
    print("=" * 50)
    
    try:
        # Import the orchestrator
        from iace_v2_orchestrator import QECSOrchestrator
        
        # Create orchestrator instance
        orchestrator = QECSOrchestrator()
        
        # Start telemetry streaming
        print("Starting telemetry streaming...")
        orchestrator.start_telemetry_streaming()
        
        # Run orchestration phases
        print("\n[INIT] Starting QECS v2.0 orchestration...")
        orchestrator.phase_i_core_system()
        time.sleep(1)  # Small delay between phases
        
        orchestrator.phase_ii_iii_transaction_security()
        time.sleep(1)  # Small delay between phases
        
        # Example cluster IDs; in production, dynamically discovered
        cluster_ids = ["C-01", "C-02", "C-03"]
        orchestrator.run_gravity_well_daemon(cluster_ids)
        time.sleep(1)  # Small delay between phases
        
        orchestrator.phase_iv_agi_report()
        
        # Finalize
        exit_code = orchestrator.finalize()
        
        # Stop telemetry streaming
        orchestrator.stop_telemetry_streaming()
        
        print(f"\n[COMPLETE] IACE v2.0 execution finished with exit code {exit_code}")
        return exit_code
        
    except ImportError as e:
        print(f"❌ Error importing QECS modules: {e}")
        print("Please ensure all QECS components are properly installed.")
        return 1
    except Exception as e:
        print(f"❌ Error running IACE v2.0: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())