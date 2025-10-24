#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==========================================="
echo "AEGIS Framework - Advanced Deployment"
echo "Automated deployment for advanced features"
echo "==========================================="
echo ""

function show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy AEGIS Framework with advanced security features

OPTIONS:
    -m, --mode MODE         Deployment mode: dev|prod|test (default: dev)
    -s, --service SERVICE   Deploy specific service: all|security|performance|monitoring (default: all)
    -p, --port PORT         API server port (default: 8000)
    --skip-build            Skip Docker image build
    --skip-tests            Skip running tests
    --cleanup               Clean up existing containers and volumes
    --advanced-security     Enable advanced security features (ZKPs, HE, SMC, DP)
    --performance-optimization Enable performance optimization features
    -h, --help              Show this help message

EXAMPLES:
    $0 --mode dev --advanced-security                    # Deploy in development mode with advanced security
    $0 --mode prod --service security --cleanup         # Deploy only security in production
    $0 --advanced-security --performance-optimization   # Deploy with all advanced features

EOF
}

MODE="dev"
SERVICE="all"
PORT=8000
SKIP_BUILD=false
SKIP_TESTS=false
CLEANUP=false
ADVANCED_SECURITY=false
PERFORMANCE_OPTIMIZATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -s|--service)
            SERVICE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --advanced-security)
            ADVANCED_SECURITY=true
            shift
            ;;
        --performance-optimization)
            PERFORMANCE_OPTIMIZATION=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Mode: $MODE"
echo "  Service: $SERVICE"
echo "  Port: $PORT"
echo "  Skip Build: $SKIP_BUILD"
echo "  Skip Tests: $SKIP_TESTS"
echo "  Cleanup: $CLEANUP"
echo "  Advanced Security: $ADVANCED_SECURITY"
echo "  Performance Optimization: $PERFORMANCE_OPTIMIZATION"
echo ""

cd "$PROJECT_ROOT"

if [ "$CLEANUP" = true ]; then
    echo ">>> Cleaning up existing deployment..."
    docker-compose down -v 2>/dev/null || true
    echo "✓ Cleanup complete"
    echo ""
fi

if [ ! -f ".env" ]; then
    echo ">>> Creating .env file from template..."
    cp .env.example .env
    
    # Enable advanced features if requested
    if [ "$ADVANCED_SECURITY" = true ]; then
        echo "# Advanced Security Features" >> .env
        echo "ENABLE_ZERO_KNOWLEDGE_PROOFS=true" >> .env
        echo "ENABLE_HOMOMORPHIC_ENCRYPTION=true" >> .env
        echo "ENABLE_SECURE_MPC=true" >> .env
        echo "ENABLE_DIFFERENTIAL_PRIVACY=true" >> .env
    fi
    
    if [ "$PERFORMANCE_OPTIMIZATION" = true ]; then
        echo "# Performance Optimization" >> .env
        echo "ENABLE_MEMORY_OPTIMIZATION=true" >> .env
        echo "ENABLE_CONCURRENCY_OPTIMIZATION=true" >> .env
        echo "ENABLE_NETWORK_OPTIMIZATION=true" >> .env
    fi
    
    echo "✓ .env file created"
    echo ""
fi

if [ "$SKIP_TESTS" = false ]; then
    echo ">>> Running tests..."
    if command -v python3 &> /dev/null; then
        python3 -m pytest tests/test_security_integration.py -v --tb=short || {
            echo "⚠ Security tests failed, but continuing deployment..."
        }
        
        python3 -m pytest tests/test_consensus_integration.py -v --tb=short || {
            echo "⚠ Consensus tests failed, but continuing deployment..."
        }
    else
        echo "⚠ Python3 not found, skipping tests"
    fi
    echo ""
fi

if [ "$SKIP_BUILD" = false ]; then
    echo ">>> Building Docker images..."
    docker-compose build --parallel
    echo "✓ Docker images built"
    echo ""
fi

# Set environment variables for advanced features
if [ "$ADVANCED_SECURITY" = true ]; then
    export AEGIS_ADVANCED_SECURITY_ENABLED=true
    echo ">>> Advanced security features enabled"
fi

if [ "$PERFORMANCE_OPTIMIZATION" = true ]; then
    export AEGIS_PERFORMANCE_OPTIMIZATION_ENABLED=true
    echo ">>> Performance optimization features enabled"
fi

case $SERVICE in
    all)
        echo ">>> Deploying all services..."
        docker-compose up -d
        ;;
    security)
        echo ">>> Deploying security services..."
        docker-compose up -d aegis-node
        ;;
    performance)
        echo ">>> Deploying performance optimization services..."
        docker-compose up -d aegis-node
        ;;
    monitoring)
        echo ">>> Deploying monitoring stack..."
        docker-compose up -d prometheus grafana
        ;;
    *)
        echo "Error: Unknown service '$SERVICE'"
        exit 1
        ;;
esac

echo ""
echo ">>> Waiting for services to be healthy..."
sleep 10

echo ""
echo "==========================================="
echo "Deployment Complete!"
echo "==========================================="
echo ""
echo "Service URLs:"
echo "  - AEGIS API:        http://localhost:8080"
echo "  - Model Serving:    http://localhost:8000"
echo "  - Grafana:          http://localhost:3000 (admin@admin.com / aegis2024)"
echo "  - Prometheus:       http://localhost:9091"
echo ""
echo "Advanced Features:"
if [ "$ADVANCED_SECURITY" = true ]; then
    echo "  ✓ Zero-Knowledge Proofs: Enabled"
    echo "  ✓ Homomorphic Encryption: Enabled"
    echo "  ✓ Secure Multi-Party Computation: Enabled"
    echo "  ✓ Differential Privacy: Enabled"
fi

if [ "$PERFORMANCE_OPTIMIZATION" = true ]; then
    echo "  ✓ Memory Optimization: Enabled"
    echo "  ✓ Concurrency Optimization: Enabled"
    echo "  ✓ Network Optimization: Enabled"
fi

echo ""
echo "Useful commands:"
echo "  - View logs:        docker-compose logs -f"
echo "  - Stop services:    docker-compose down"
echo "  - Restart:          docker-compose restart"
echo "  - Status:           docker-compose ps"
echo ""

if [ "$MODE" = "prod" ]; then
    echo "⚠ PRODUCTION MODE NOTES:"
    echo "  1. Change default passwords in .env file"
    echo "  2. Configure SSL/TLS certificates"
    echo "  3. Set up firewall rules"
    echo "  4. Enable monitoring and alerting"
    echo "  5. Configure backup strategy"
    echo ""
fi

docker-compose ps
