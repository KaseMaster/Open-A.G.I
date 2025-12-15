#!/usr/bin/env python3
"""
Simple Production Deployment Script for Harmonic Mesh Network (HMN) Nodes
This script deploys multiple HMN nodes directly using Python without Docker
"""

import os
import sys
import subprocess
import time
import json
import argparse
import threading
import signal
import atexit
from typing import List, Dict, Any, Optional

# Global list to keep track of running processes
running_processes = []

def cleanup_processes():
    """Clean up all running processes on exit"""
    print("\n🛑 Cleaning up running processes...")
    for process in running_processes:
        try:
            process.terminate()
            process.wait(timeout=5)
            print(f"✅ Terminated process {process.pid}")
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"❌ Force killed process {process.pid}")
        except Exception as e:
            print(f"❌ Error terminating process {process.pid}: {e}")

def create_node_config(node_id: str, ports: Dict[str, int]) -> Dict[str, Any]:
    """Create configuration for an HMN node"""
    return {
        "node_id": node_id,
        "network_config": {
            "shard_count": 10,
            "replication_factor": 3,
            "validator_count": 5,
            "network_peers": [],
            "metrics_port": ports["metrics"],
            "service_ports": ports
        },
        "data_directory": f"./hmn-data/{node_id}",
        "log_level": "INFO"
    }

def initialize_node(node_id: str, config: Dict[str, Any]) -> bool:
    """Initialize an HMN node with the given configuration"""
    try:
        # Create data directory
        data_dir = f"./hmn-data/{node_id}"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save configuration
        config_file = f"./hmn-data/{node_id}/config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Node {node_id} initialized with config: {config_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize node {node_id}: {e}")
        return False

def start_node(node_id: str) -> Optional[subprocess.Popen]:
    """Start an HMN node as a subprocess"""
    try:
        # Start the node as a subprocess
        process = subprocess.Popen([
            sys.executable, "-m", "src.network.hmn.deploy_node", 
            "start", "--config", f"./hmn-data/{node_id}/config.json"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Add to running processes list
        running_processes.append(process)
        
        print(f"🚀 Node {node_id} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start node {node_id}: {e}")
        return None

def check_node_health(node_id: str, port: int) -> bool:
    """Check the health of an HMN node"""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/metrics", timeout=5)
        return response.status_code == 200
    except Exception as e:
        # print(f"❌ Health check failed for node {node_id}: {e}")
        return False

def deploy_hmn_cluster(node_count: int = 3) -> List[subprocess.Popen]:
    """Deploy a cluster of HMN nodes"""
    print(f"🚀 Deploying HMN cluster with {node_count} nodes...")
    
    # Create necessary directories
    os.makedirs("./hmn-data", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Base ports
    base_ports = {
        "metrics": 8000,
        "ledger": 8001,
        "cal_engine": 8002,
        "mining": 8003,
        "memory_mesh": 8004,
        "consensus": 8005
    }
    
    processes: List[subprocess.Popen] = []
    
    # Initialize and start nodes
    for i in range(1, node_count + 1):
        node_id = f"hmn-node-{i:03d}"
        
        # Adjust ports for each node
        ports = {k: v + (i-1) * 10 for k, v in base_ports.items()}
        
        # Create configuration
        config = create_node_config(node_id, ports)
        
        # Initialize node
        if not initialize_node(node_id, config):
            print(f"❌ Failed to initialize node {node_id}")
            continue
        
        # Start node
        process = start_node(node_id)
        if process:
            processes.append(process)
            
            # Wait a bit between starting nodes
            time.sleep(2)
    
    print(f"📊 Started {len(processes)} HMN nodes")
    return processes

def monitor_cluster(processes: List[subprocess.Popen]):
    """Monitor the HMN cluster"""
    print("👀 Monitoring HMN cluster...")
    
    try:
        while True:
            healthy_nodes = 0
            running_nodes = 0
            
            for i, process in enumerate(processes):
                node_id = f"hmn-node-{i+1:03d}"
                if process.poll() is None:  # Process is still running
                    running_nodes += 1
                    port = 8000 + i * 10
                    if check_node_health(node_id, port):
                        healthy_nodes += 1
                        print(f"✅ Node {node_id} is healthy")
                    else:
                        print(f"⚠️ Node {node_id} is running but not responding")
                else:
                    print(f"❌ Node process {node_id} has terminated with code {process.returncode}")
            
            print(f"📊 Cluster Health: {healthy_nodes}/{len(processes)} nodes healthy ({running_nodes} running)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping cluster monitoring...")
        return

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy HMN to production')
    parser.add_argument('--nodes', type=int, default=3, help='Number of HMN nodes to deploy')
    parser.add_argument('--monitor', action='store_true', help='Monitor cluster after deployment')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 HMN Production Deployment")
    print("=" * 60)
    
    # Register cleanup function
    atexit.register(cleanup_processes)
    
    # Deploy cluster
    processes = deploy_hmn_cluster(args.nodes)
    
    if not processes:
        print("❌ No nodes were successfully deployed")
        return False
    
    print("\n" + "=" * 60)
    print("✅ HMN Production Deployment Complete!")
    print("=" * 60)
    print(f"📊 Deployed {len(processes)} HMN nodes")
    print("🌐 Access nodes on ports 8000-8005 (with offsets)")
    print("📈 Monitor metrics at http://localhost:8000/metrics (and similar for other nodes)")
    
    # Monitor if requested
    if args.monitor:
        monitor_cluster(processes)
    else:
        # Just wait for user input to keep processes running
        print("\n⏳ Nodes are running in the background...")
        print("Press Ctrl+C to stop all nodes")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all nodes...")
            cleanup_processes()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)