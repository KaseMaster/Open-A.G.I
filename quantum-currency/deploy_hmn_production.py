#!/usr/bin/env python3
"""
Production Deployment Script for Harmonic Mesh Network (HMN) Nodes
"""

import os
import sys
import subprocess
import time
import json
import argparse
from typing import List, Dict, Any

def check_docker_installed():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not installed or not in PATH")
        return False

def check_docker_compose_installed():
    """Check if Docker Compose is installed"""
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker Compose is not installed or not in PATH")
        return False

def create_hmn_directories():
    """Create necessary directories for HMN nodes"""
    directories = [
        "hmn-data/node1",
        "hmn-data/node2",
        "hmn-data/node3",
        "logs",
        "data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def build_hmn_images():
    """Build HMN Docker images"""
    print("🏗️ Building HMN Docker images...")
    try:
        # Build the HMN node image
        subprocess.run([
            'docker', 'build', 
            '-t', 'quantum-currency/hmn-node:latest',
            '-f', 'Dockerfile.hmn-node',
            '.'
        ], check=True, capture_output=True)
        print("✅ HMN node image built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build HMN node image: {e}")
        return False

def deploy_hmn_cluster(node_count: int = 3):
    """Deploy HMN cluster using Docker Compose"""
    print(f"🚀 Deploying HMN cluster with {node_count} nodes...")
    
    # Create docker-compose override file for the specified number of nodes
    compose_override = {
        "version": "3.8",
        "services": {}
    }
    
    # Add HMN nodes
    for i in range(1, node_count + 1):
        node_name = f"hmn-node-{i}"
        compose_override["services"][node_name] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile.hmn-node"
            },
            "container_name": node_name,
            "restart": "unless-stopped",
            "environment": [
                f"PYTHONPATH=/app/src",
                f"NODE_ID={node_name}"
            ],
            "ports": [
                f"800{i}:8001",
                f"801{i}:8002",
                f"802{i}:8003",
                f"803{i}:8004",
                f"804{i}:8005"
            ],
            "volumes": [
                f"./hmn-data/node{i}:/app/data",
                "./logs:/var/log/quantum"
            ],
            "networks": ["quantum-network"],
            "depends_on": ["agi-coordinator"]
        }
    
    # Write override file
    with open('docker-compose.hmn-override.yml', 'w') as f:
        json.dump(compose_override, f, indent=2)
    
    try:
        # Deploy using docker-compose
        subprocess.run([
            'docker-compose',
            '-f', 'docker-compose.yml',
            '-f', 'docker-compose.hmn-override.yml',
            'up', '-d'
        ], check=True)
        print("✅ HMN cluster deployed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to deploy HMN cluster: {e}")
        return False

def check_hmn_health():
    """Check the health of deployed HMN nodes"""
    print("🏥 Checking HMN node health...")
    
    try:
        # Get running containers
        result = subprocess.run([
            'docker', 'ps', '--format', '{{.Names}}'
        ], capture_output=True, text=True, check=True)
        
        containers = result.stdout.strip().split('\n')
        hmn_containers = [c for c in containers if c.startswith('hmn-node-')]
        
        if not hmn_containers:
            print("❌ No HMN containers found")
            return False
        
        healthy_nodes = 0
        for container in hmn_containers:
            # Check container health
            result = subprocess.run([
                'docker', 'inspect', container, '--format', '{{.State.Status}}'
            ], capture_output=True, text=True, check=True)
            
            status = result.stdout.strip()
            if status == 'running':
                print(f"✅ {container} is running")
                healthy_nodes += 1
            else:
                print(f"❌ {container} is {status}")
        
        print(f"📊 HMN Health Check: {healthy_nodes}/{len(hmn_containers)} nodes healthy")
        return healthy_nodes == len(hmn_containers)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to check HMN health: {e}")
        return False

def get_hmn_metrics():
    """Get metrics from HMN nodes"""
    print("📈 Getting HMN metrics...")
    
    try:
        # For demonstration, we'll just show how to get logs
        result = subprocess.run([
            'docker', 'logs', 'hmn-node-1', '--tail', '10'
        ], capture_output=True, text=True, check=True)
        
        print("📋 Recent logs from hmn-node-1:")
        print(result.stdout[-500:])  # Last 500 characters
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get HMN metrics: {e}")
        return False

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy HMN to production')
    parser.add_argument('--nodes', type=int, default=3, help='Number of HMN nodes to deploy')
    parser.add_argument('--check-health', action='store_true', help='Check health after deployment')
    parser.add_argument('--get-metrics', action='store_true', help='Get metrics after deployment')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 HMN Production Deployment")
    print("=" * 60)
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    if not check_docker_installed():
        return False
    
    if not check_docker_compose_installed():
        return False
    
    # Create directories
    print("\n📂 Creating directories...")
    create_hmn_directories()
    
    # Build images
    print("\n🏗️ Building Docker images...")
    if not build_hmn_images():
        return False
    
    # Deploy cluster
    print("\n🚀 Deploying HMN cluster...")
    if not deploy_hmn_cluster(args.nodes):
        return False
    
    # Check health if requested
    if args.check_health:
        print("\n🏥 Checking cluster health...")
        time.sleep(10)  # Wait for containers to start
        if not check_hmn_health():
            return False
    
    # Get metrics if requested
    if args.get_metrics:
        print("\n📈 Getting cluster metrics...")
        time.sleep(5)  # Wait for services to initialize
        get_hmn_metrics()
    
    print("\n" + "=" * 60)
    print("✅ HMN Production Deployment Complete!")
    print("=" * 60)
    print(f"📊 Deployed {args.nodes} HMN nodes")
    print("🌐 Access nodes on ports 8001-8005 (with offsets)")
    print("📈 Monitor metrics at http://localhost:9090 (Prometheus)")
    print("📊 View dashboards at http://localhost:3000 (Grafana)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)