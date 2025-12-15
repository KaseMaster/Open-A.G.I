#!/usr/bin/env python3
"""
Deployment Script for Harmonic Mesh Network Full Node
Provides easy deployment and configuration of HMN nodes
"""

import asyncio
import argparse
import json
import os
import sys
import time
from typing import Dict, Any
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from network.hmn.full_node import FullNode

def load_node_config(config_file: str) -> Dict[str, Any]:
    """Load node configuration from file"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Return default configuration
        return {
            "shard_count": 10,
            "replication_factor": 3,
            "validator_count": 5,
            "network_peers": [],
            "data_directory": "./data",
            "log_level": "INFO",
            "metrics_port": 8000
        }

def save_node_config(config: Dict[str, Any], config_file: str):
    """Save node configuration to file"""
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def create_default_config(config_file: str):
    """Create a default configuration file"""
    default_config = {
        "node_id": "hmn-node-001",
        "network_config": {
            "shard_count": 10,
            "replication_factor": 3,
            "validator_count": 5,
            "network_peers": [
                "hmn-node-002:8000",
                "hmn-node-003:8000",
                "hmn-node-004:8000",
                "hmn-node-005:8000"
            ],
            "metrics_port": 8000
        },
        "data_directory": "./data",
        "log_level": "INFO",
        "service_ports": {
            "ledger": 8001,
            "cal_engine": 8002,
            "mining_agent": 8003,
            "memory_mesh": 8004,
            "consensus": 8005
        }
    }
    
    save_node_config(default_config, config_file)
    print(f"Created default configuration file: {config_file}")

async def start_node(config_file: str):
    """Start a full node with the given configuration"""
    print("🚀 Starting Harmonic Mesh Network Full Node")
    print("=" * 45)
    
    # Load configuration
    config = load_node_config(config_file)
    node_id = config.get("node_id", "hmn-node-001")
    network_config = config.get("network_config", {})
    
    print(f"Node ID: {node_id}")
    print(f"Configuration: {config_file}")
    
    # Create and start node
    node = FullNode(node_id, network_config)
    
    # Add sample validators if none exist
    if not node.consensus_engine.validators:
        validators_data = [
            ("validator-1", 0.95, 10000.0),
            ("validator-2", 0.87, 8000.0),
            ("validator-3", 0.75, 12000.0),
            ("validator-4", 0.92, 9000.0),
            ("validator-5", 0.85, 7000.0),
        ]
        
        for validator_id, psi_score, stake in validators_data:
            node.consensus_engine.add_validator(validator_id, psi_score, stake)
        
        print(f"✅ Added {len(validators_data)} sample validators")
    
    # Start node services
    print("🔄 Starting node services...")
    try:
        await node.run_node_services()
    except KeyboardInterrupt:
        print("\n👋 Node services stopped by user")
    except Exception as e:
        print(f"❌ Error running node services: {e}")
    finally:
        node.stop()

def show_node_status(config_file: str):
    """Show the current status of a node"""
    print("📊 Node Status")
    print("=" * 15)
    
    # Load configuration
    config = load_node_config(config_file)
    node_id = config.get("node_id", "hmn-node-001")
    
    print(f"Node ID: {node_id}")
    print(f"Configuration: {config_file}")
    print(f"Status: {'Running' if os.path.exists('.node_running') else 'Stopped'}")

def show_node_health(config_file: str):
    """Show detailed health status of a node"""
    print("🩺 Node Health Check")
    print("=" * 20)
    
    # Load configuration
    config = load_node_config(config_file)
    node_id = config.get("node_id", "hmn-node-001")
    network_config = config.get("network_config", {})
    
    print(f"Node ID: {node_id}")
    print(f"Configuration: {config_file}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create a temporary node instance to check health
    # In a real implementation, this would connect to a running node
    node = FullNode(node_id, network_config)
    
    # Add sample validators for demonstration
    validators_data = [
        ("validator-1", 0.95, 10000.0),
        ("validator-2", 0.87, 8000.0),
        ("validator-3", 0.75, 12000.0),
        ("validator-4", 0.92, 9000.0),
        ("validator-5", 0.85, 7000.0),
    ]
    
    for validator_id, psi_score, stake in validators_data:
        node.consensus_engine.add_validator(validator_id, psi_score, stake)
    
    # Get health status
    health = node.get_health_status()
    
    print(f"\nOverall Health: {'✅ Healthy' if health['overall_health'] else '❌ Unhealthy'}")
    
    print("\nService Health:")
    for service, status in health['services'].items():
        health_indicator = "✅" if status['healthy'] else "❌"
        print(f"  {health_indicator} {service}: {'Healthy' if status['healthy'] else 'Unhealthy'}")
        if not status['healthy']:
            print(f"      Last Error: {status['error']}")
            print(f"      Restarts: {status['restart_count']}")
        print(f"      Last Check: {datetime.fromtimestamp(status['last_check']).strftime('%H:%M:%S')}")

def show_node_stats(config_file: str):
    """Show detailed statistics of a node"""
    print("📈 Node Statistics")
    print("=" * 18)
    
    # Load configuration
    config = load_node_config(config_file)
    node_id = config.get("node_id", "hmn-node-001")
    network_config = config.get("network_config", {})
    
    print(f"Node ID: {node_id}")
    print(f"Configuration: {config_file}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create a temporary node instance to get stats
    # In a real implementation, this would connect to a running node
    node = FullNode(node_id, network_config)
    
    # Add sample validators for demonstration
    validators_data = [
        ("validator-1", 0.95, 10000.0),
        ("validator-2", 0.87, 8000.0),
        ("validator-3", 0.75, 12000.0),
        ("validator-4", 0.92, 9000.0),
        ("validator-5", 0.85, 7000.0),
    ]
    
    for validator_id, psi_score, stake in validators_data:
        node.consensus_engine.add_validator(validator_id, psi_score, stake)
    
    # Get node stats
    stats = node.get_node_stats()
    
    print(f"\nNode Status: {'Running' if stats['running'] else 'Stopped'}")
    print(f"Health Status: {'Healthy' if stats['health_status'] else 'Unhealthy'}")
    
    print("\nCAL Engine State:")
    cal_state = stats['cal_state']
    print(f"  λ(t): {cal_state['lambda_t']:.3f}")
    print(f"  Ĉ(t): {cal_state['coherence_density']:.3f}")
    print(f"  Ψ: {cal_state['psi_score']:.3f}")
    
    print("\nMemory Mesh Stats:")
    memory_stats = stats['memory_stats']
    print(f"  Local Updates: {memory_stats['local_updates_count']}")
    print(f"  Archived Updates: {memory_stats['archived_updates_count']}")
    print(f"  Connected Peers: {memory_stats['connected_peers']}")
    
    print("\nConsensus Stats:")
    consensus_stats = stats['consensus_stats']
    print(f"  Validators: {consensus_stats['validators_count']}")
    print(f"  Consensus Rounds: {consensus_stats['consensus_history_count']}")
    
    print(f"\nMining Agent:")
    print(f"  Epoch Count: {stats['mining_epoch_count']}")
    
    print(f"\nLayer 1 Ledger:")
    print(f"  Transactions: {stats['ledger_transaction_count']}")
    
    print("\nService Health:")
    service_health = stats['service_health']
    for service, health in service_health.items():
        health_indicator = "✅" if health['healthy'] else "❌"
        print(f"  {health_indicator} {service}: {'Healthy' if health['healthy'] else 'Unhealthy'}")
        if not health['healthy']:
            print(f"      Error: {health['last_error']}")
        print(f"      Restarts: {health['restarts']}")

def main():
    """Main entry point for the deployment script"""
    parser = argparse.ArgumentParser(description="Harmonic Mesh Network Node Deployment")
    parser.add_argument("action", choices=["start", "stop", "status", "init", "health", "stats"], 
                       help="Action to perform")
    parser.add_argument("--config", "-c", default="node_config.json",
                       help="Configuration file (default: node_config.json)")
    parser.add_argument("--node-id", "-n", help="Node ID (overrides config file)")
    
    args = parser.parse_args()
    
    # Handle actions
    if args.action == "init":
        create_default_config(args.config)
        return
    
    if args.action == "status":
        show_node_status(args.config)
        return
    
    if args.action == "health":
        show_node_health(args.config)
        return
    
    if args.action == "stats":
        show_node_stats(args.config)
        return
    
    if args.action == "start":
        # Check if config file exists, create default if not
        if not os.path.exists(args.config):
            print(f"Configuration file {args.config} not found. Creating default...")
            create_default_config(args.config)
        
        # Run the node
        try:
            asyncio.run(start_node(args.config))
        except KeyboardInterrupt:
            print("\n👋 Deployment script stopped by user")
        return
    
    if args.action == "stop":
        print("🛑 Stopping node...")
        # In a real implementation, this would send a signal to stop the running node
        if os.path.exists(".node_running"):
            os.remove(".node_running")
        print("✅ Node stopped")
        return

if __name__ == "__main__":
    main()