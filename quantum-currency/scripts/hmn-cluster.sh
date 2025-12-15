#!/bin/bash

# HMN Cluster Management Script
# Provides easy deployment and management of HMN node clusters

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start-cluster     Start a multi-node HMN cluster"
    echo "  stop-cluster      Stop the HMN cluster"
    echo "  scale-cluster N   Scale cluster to N nodes"
    echo "  cluster-status    Show cluster status"
    echo "  cluster-logs      Show cluster logs"
    echo "  help             Show this help message"
    echo ""
}

start_cluster() {
    echo "🚀 Starting HMN Cluster"
    echo "======================"
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/hmn-data/node1" "$PROJECT_ROOT/hmn-data/node2" "$PROJECT_ROOT/hmn-data/node3"
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Start cluster using docker-compose
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" up -d hmn-node-1 hmn-node-2
    
    echo "✅ HMN Cluster started with 2 nodes"
    echo "   Node 1: http://localhost:8001"
    echo "   Node 2: http://localhost:8011"
    echo "   Metrics: http://localhost:8000/metrics"
}

stop_cluster() {
    echo "🛑 Stopping HMN Cluster"
    echo "======================"
    
    # Stop cluster using docker-compose
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" stop hmn-node-1 hmn-node-2
    
    echo "✅ HMN Cluster stopped"
}

scale_cluster() {
    local node_count=$1
    
    if [[ -z "$node_count" ]]; then
        echo "Error: Node count required"
        usage
        exit 1
    fi
    
    if ! [[ "$node_count" =~ ^[0-9]+$ ]]; then
        echo "Error: Node count must be a number"
        exit 1
    fi
    
    echo "🔄 Scaling HMN Cluster to $node_count nodes"
    echo "========================================="
    
    # Stop existing cluster
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" stop hmn-node-1 hmn-node-2
    
    # Scale to requested number of nodes
    for i in $(seq 1 $node_count); do
        # Create data directory
        mkdir -p "$PROJECT_ROOT/hmn-data/node$i"
        
        # Start node
        NODE_ID="hmn-node-$i" docker-compose -f "$PROJECT_ROOT/docker-compose.yml" up -d "hmn-node-$i"
    done
    
    echo "✅ HMN Cluster scaled to $node_count nodes"
}

cluster_status() {
    echo "📊 HMN Cluster Status"
    echo "===================="
    
    # Show docker-compose status for HMN nodes
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps hmn-node-1 hmn-node-2
}

cluster_logs() {
    echo "📋 HMN Cluster Logs"
    echo "=================="
    
    # Show logs for HMN nodes
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" logs -f hmn-node-1 hmn-node-2
}

# Main script logic
case "$1" in
    start-cluster)
        start_cluster
        ;;
    stop-cluster)
        stop_cluster
        ;;
    scale-cluster)
        scale_cluster "$2"
        ;;
    cluster-status)
        cluster_status
        ;;
    cluster-logs)
        cluster_logs
        ;;
    help|"")
        usage
        ;;
    *)
        echo "Error: Unknown command '$1'"
        usage
        exit 1
        ;;
esac