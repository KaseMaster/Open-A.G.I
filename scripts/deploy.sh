#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==================================="
echo "AEGIS Framework - Deployment Script"
echo "==================================="
echo ""

function show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy AEGIS Framework with various configurations

OPTIONS:
    -m, --mode MODE         Deployment mode: dev|prod|test (default: dev)
    -s, --service SERVICE   Deploy specific service: all|api|monitoring|ml (default: all)
    -p, --port PORT         API server port (default: 8000)
    --skip-build            Skip Docker image build
    --skip-tests            Skip running tests
    --cleanup               Clean up existing containers and volumes
    -h, --help              Show this help message

EXAMPLES:
    $0 --mode dev                    # Deploy in development mode
    $0 --mode prod --service api     # Deploy only API in production
    $0 --cleanup --mode dev          # Clean up and deploy fresh

EOF
}

MODE="dev"
SERVICE="all"
PORT=8000
SKIP_BUILD=false
SKIP_TESTS=false
CLEANUP=false

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
    echo "✓ .env file created"
    echo ""
fi

if [ "$SKIP_TESTS" = false ]; then
    echo ">>> Running tests..."
    if command -v python3 &> /dev/null; then
        python3 -m pytest tests/ -v --tb=short || {
            echo "⚠ Tests failed, but continuing deployment..."
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

case $SERVICE in
    all)
        echo ">>> Deploying all services..."
        docker-compose up -d
        ;;
    api)
        echo ">>> Deploying API service..."
        docker-compose up -d aegis-node
        ;;
    monitoring)
        echo ">>> Deploying monitoring stack..."
        docker-compose up -d prometheus grafana
        ;;
    ml)
        echo ">>> Deploying ML services..."
        docker-compose up -d aegis-node
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
echo "==================================="
echo "Deployment Complete!"
echo "==================================="
echo ""
echo "Service URLs:"
echo "  - AEGIS API:        http://localhost:8080"
echo "  - Model Serving:    http://localhost:8000"
echo "  - Grafana:          http://localhost:3000 (admin / aegis2024)"
echo "  - Prometheus:       http://localhost:9091"
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
