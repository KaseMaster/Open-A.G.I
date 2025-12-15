#!/usr/bin/env python3
"""
Main entry point for executing the AFIP (Absolute Field Integrity Protocol) v1.0
"""

import argparse
import json
import sys
import os
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from afip.orchestrator import AFIPOrchestrator

def load_nodes_from_file(filepath: str) -> List[Dict[str, Any]]:
    """Load node configurations from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Node configuration file {filepath} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        return []

def load_telemetry_from_file(filepath: str) -> List[Dict[str, Any]]:
    """Load telemetry data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Telemetry data file {filepath} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Execute AFIP (Absolute Field Integrity Protocol) v1.0')
    parser.add_argument('--nodes', type=str, help='Path to JSON file containing node configurations')
    parser.add_argument('--telemetry', type=str, help='Path to JSON file containing telemetry data')
    parser.add_argument('--config', type=str, help='Path to JSON file containing AFIP configuration')
    parser.add_argument('--shard-count', type=int, default=5, help='Number of shards to create')
    parser.add_argument('--observation-days', type=int, default=7, help='Observation period in days')
    parser.add_argument('--tee-enabled', action='store_true', help='Enable Trusted Execution Environment')
    parser.add_argument('--output', type=str, help='Path to save execution report')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            config = {}
    else:
        config = {}
    
    # Override config with command line arguments
    config.setdefault("shard_count", args.shard_count)
    config.setdefault("observation_period_days", args.observation_days)
    config.setdefault("tee_enabled", args.tee_enabled)
    
    # Initialize AFIP orchestrator
    print("⚛️ Initializing AFIP v1.0 Orchestrator")
    afip = AFIPOrchestrator(config)
    
    # Load nodes
    if args.nodes:
        nodes = load_nodes_from_file(args.nodes)
        if not nodes:
            print("No nodes loaded, using default example nodes")
            nodes = [
                {"node_id": "node_001", "coherence_score": 0.98, 
                 "qra_params": {"n": 1, "l": 0, "m": 0, "s": 0.5}},
                {"node_id": "node_002", "coherence_score": 0.96,
                 "qra_params": {"n": 2, "l": 1, "m": -1, "s": 0.7}},
                {"node_id": "node_003", "coherence_score": 0.97,
                 "qra_params": {"n": 1, "l": 1, "m": 0, "s": 0.3}},
            ]
    else:
        # Default example nodes
        nodes = [
            {"node_id": "node_001", "coherence_score": 0.98, 
             "qra_params": {"n": 1, "l": 0, "m": 0, "s": 0.5}},
            {"node_id": "node_002", "coherence_score": 0.96,
             "qra_params": {"n": 2, "l": 1, "m": -1, "s": 0.7}},
            {"node_id": "node_003", "coherence_score": 0.97,
             "qra_params": {"n": 1, "l": 1, "m": 0, "s": 0.3}},
        ]
    
    # Load telemetry data
    if args.telemetry:
        telemetry_data = load_telemetry_from_file(args.telemetry)
        if not telemetry_data:
            print("No telemetry data loaded, using default example data")
            telemetry_data = [
                {"node_id": "node_001", "g_vector_magnitude": 0.5, "coherence": 0.98, "rsi": 0.92, "delta_h": 0.001},
                {"node_id": "node_001", "g_vector_magnitude": 0.7, "coherence": 0.97, "rsi": 0.91, "delta_h": 0.002},
                {"node_id": "node_001", "g_vector_magnitude": 0.9, "coherence": 0.96, "rsi": 0.93, "delta_h": 0.001},
                {"node_id": "node_002", "g_vector_magnitude": 0.4, "coherence": 0.96, "rsi": 0.94, "delta_h": 0.0005},
                {"node_id": "node_002", "g_vector_magnitude": 0.6, "coherence": 0.95, "rsi": 0.92, "delta_h": 0.001},
                {"node_id": "node_003", "g_vector_magnitude": 0.3, "coherence": 0.97, "rsi": 0.95, "delta_h": 0.0002},
            ] * 10  # Repeat for more data
    else:
        # Default example telemetry data
        telemetry_data = [
            {"node_id": "node_001", "g_vector_magnitude": 0.5, "coherence": 0.98, "rsi": 0.92, "delta_h": 0.001},
            {"node_id": "node_001", "g_vector_magnitude": 0.7, "coherence": 0.97, "rsi": 0.91, "delta_h": 0.002},
            {"node_id": "node_001", "g_vector_magnitude": 0.9, "coherence": 0.96, "rsi": 0.93, "delta_h": 0.001},
            {"node_id": "node_002", "g_vector_magnitude": 0.4, "coherence": 0.96, "rsi": 0.94, "delta_h": 0.0005},
            {"node_id": "node_002", "g_vector_magnitude": 0.6, "coherence": 0.95, "rsi": 0.92, "delta_h": 0.001},
            {"node_id": "node_003", "g_vector_magnitude": 0.3, "coherence": 0.97, "rsi": 0.95, "delta_h": 0.0002},
        ] * 10  # Repeat for more data
    
    # Execute full AFIP protocol
    print("⚛️ Executing Complete AFIP v1.0 Protocol")
    print("=" * 50)
    
    final_report = afip.execute_full_afip_protocol(nodes, telemetry_data)
    
    # Save report if output path specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(final_report, f, indent=2)
            print(f"✅ Execution report saved to {args.output}")
        except Exception as e:
            print(f"Error saving report: {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("AFIP EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Overall Success: {'✅' if final_report['overall_success'] else '❌'}")
    print(f"Final Status Code: {final_report['final_status_code']}")
    print(f"Execution Time: {final_report['total_execution_time']:.2f} seconds")
    print(f"Phase I Success: {'✅' if final_report['phase_i_success'] else '❌'}")
    print(f"Phase II Success: {'✅' if final_report['phase_ii_success'] else '❌'}")
    print(f"CI/CD Success: {'✅' if final_report['ci_cd_success'] else '❌'}")
    print(f"Phase III Success: {'✅' if final_report['phase_iii_success'] else '❌'}")
    
    if final_report['overall_success']:
        print("\n🎉 QECS IS PRODUCTION READY! 🎉")
        return 0
    else:
        print("\n⚠️ QECS REQUIRES ADDITIONAL TUNING ⚠️")
        return 1

if __name__ == "__main__":
    sys.exit(main())