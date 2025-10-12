#!/bin/bash

# AEGIS Framework - Deployment Script
# Advanced Encrypted Governance and Intelligence System
# This script handles deployment to various environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
AEGIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_REGISTRY="ghcr.io/aegis"
IMAGE_NAME="aegis-framework"
VERSION=$(cat "$AEGIS_DIR/VERSION" 2>/dev/null || echo "1.0.0")
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Default values
ENVIRONMENT="production"
BUILD_ONLY=false
PUSH_IMAGES=false
SKIP_TESTS=false
SKIP_BACKUP=false
ROLLBACK=false
HEALTH_CHECK_TIMEOUT=300

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}[AEGIS DEPLOY]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Show help
show_help() {
    cat << EOF
AEGIS Framework Deployment Script

Usage: $0 [OPTIONS] [ENVIRONMENT]

ENVIRONMENTS:
    local       Deploy to local Docker environment
    staging     Deploy to staging environment
    production  Deploy to production environment
    kubernetes  Deploy to Kubernetes cluster

OPTIONS:
    -b, --build-only        Only build images, don't deploy
    -p, --push              Push images to registry
    -t, --skip-tests        Skip running tests before deployment
    -s, --skip-backup       Skip database backup before deployment
    -r, --rollback          Rollback to previous version
    -v, --version VERSION   Specify version to deploy
    -h, --help              Show this help message

EXAMPLES:
    $0 local                    # Deploy to local environment
    $0 production --push        # Deploy to production and push images
    $0 staging --skip-tests     # Deploy to staging without tests
    $0 --rollback production    # Rollback production deployment

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--build-only)
                BUILD_ONLY=true
                shift
                ;;
            -p|--push)
                PUSH_IMAGES=true
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -s|--skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            -r|--rollback)
                ROLLBACK=true
                shift
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            local|staging|production|kubernetes)
                ENVIRONMENT="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_header "Checking deployment prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose")
    
    if [[ "$ENVIRONMENT" == "kubernetes" ]]; then
        required_commands+=("kubectl" "helm")
    fi
    
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            log_error "$cmd is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$AEGIS_DIR/.env" ]]; then
        log_error ".env file not found. Run setup script first."
        exit 1
    fi
    
    # Load environment variables
    set -a
    source "$AEGIS_DIR/.env"
    set +a
    
    log_success "Prerequisites check completed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log_warning "Skipping tests as requested"
        return
    fi
    
    log_header "Running tests before deployment..."
    
    cd "$AEGIS_DIR"
    
    # Activate virtual environment if it exists
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi
    
    # Run unit tests
    if command_exists pytest; then
        log_info "Running unit tests..."
        pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
    fi
    
    # Run security tests
    if command_exists bandit; then
        log_info "Running security tests..."
        bandit -r src/ -f json -o security-report.json || true
    fi
    
    # Run linting
    if command_exists ruff; then
        log_info "Running code quality checks..."
        ruff check src/
    fi
    
    log_success "Tests completed successfully"
}

# Build Docker images
build_images() {
    log_header "Building Docker images..."
    
    cd "$AEGIS_DIR"
    
    # Build main application image
    log_info "Building main application image..."
    docker build \
        --target production \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="$VERSION" \
        -t "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION" \
        -t "$DOCKER_REGISTRY/$IMAGE_NAME:latest" \
        .
    
    # Build development image if needed
    if [[ "$ENVIRONMENT" == "local" ]]; then
        log_info "Building development image..."
        docker build \
            --target development \
            --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
            --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
            --build-arg VERSION="$VERSION" \
            -t "$DOCKER_REGISTRY/$IMAGE_NAME:dev" \
            .
    fi
    
    log_success "Docker images built successfully"
}

# Push images to registry
push_images() {
    if [[ "$PUSH_IMAGES" != true ]]; then
        log_info "Skipping image push"
        return
    fi
    
    log_header "Pushing images to registry..."
    
    # Login to registry if credentials are available
    if [[ -n "${DOCKER_REGISTRY_USERNAME:-}" && -n "${DOCKER_REGISTRY_PASSWORD:-}" ]]; then
        echo "$DOCKER_REGISTRY_PASSWORD" | docker login "$DOCKER_REGISTRY" -u "$DOCKER_REGISTRY_USERNAME" --password-stdin
    fi
    
    # Push images
    docker push "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION"
    docker push "$DOCKER_REGISTRY/$IMAGE_NAME:latest"
    
    if [[ "$ENVIRONMENT" == "local" ]]; then
        docker push "$DOCKER_REGISTRY/$IMAGE_NAME:dev"
    fi
    
    log_success "Images pushed to registry"
}

# Create backup
create_backup() {
    if [[ "$SKIP_BACKUP" == true ]]; then
        log_warning "Skipping backup as requested"
        return
    fi
    
    log_header "Creating backup before deployment..."
    
    local backup_dir="$AEGIS_DIR/backups/$TIMESTAMP"
    mkdir -p "$backup_dir"
    
    # Backup database
    if [[ -n "${POSTGRES_HOST:-}" ]]; then
        log_info "Backing up PostgreSQL database..."
        PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            --no-password \
            --verbose \
            --clean \
            --if-exists \
            > "$backup_dir/postgres_backup.sql"
    fi
    
    # Backup Redis data
    if [[ -n "${REDIS_HOST:-}" ]]; then
        log_info "Backing up Redis data..."
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --rdb "$backup_dir/redis_backup.rdb"
    fi
    
    # Backup configuration files
    log_info "Backing up configuration files..."
    cp -r "$AEGIS_DIR/config" "$backup_dir/"
    cp "$AEGIS_DIR/.env" "$backup_dir/"
    
    # Backup certificates and keys
    if [[ -d "$AEGIS_DIR/certs" ]]; then
        cp -r "$AEGIS_DIR/certs" "$backup_dir/"
    fi
    
    if [[ -d "$AEGIS_DIR/keys" ]]; then
        cp -r "$AEGIS_DIR/keys" "$backup_dir/"
    fi
    
    # Create backup manifest
    cat > "$backup_dir/manifest.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "version": "$VERSION",
    "environment": "$ENVIRONMENT",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "backup_type": "pre-deployment"
}
EOF
    
    log_success "Backup created at $backup_dir"
}

# Deploy to local environment
deploy_local() {
    log_header "Deploying to local environment..."
    
    cd "$AEGIS_DIR"
    
    # Stop existing services
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
    
    # Start services
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    
    # Wait for services to be ready
    wait_for_health_check "http://localhost:8080/health"
    
    log_success "Local deployment completed"
}

# Deploy to staging environment
deploy_staging() {
    log_header "Deploying to staging environment..."
    
    cd "$AEGIS_DIR"
    
    # Update environment variables for staging
    export AEGIS_ENV=staging
    export AEGIS_DEBUG=true
    
    # Stop existing services
    docker-compose down
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    wait_for_health_check "http://staging.aegis.local:8080/health"
    
    log_success "Staging deployment completed"
}

# Deploy to production environment
deploy_production() {
    log_header "Deploying to production environment..."
    
    # Additional safety checks for production
    if [[ "$ROLLBACK" != true ]]; then
        read -p "Are you sure you want to deploy to PRODUCTION? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_info "Production deployment cancelled"
            exit 0
        fi
    fi
    
    cd "$AEGIS_DIR"
    
    # Update environment variables for production
    export AEGIS_ENV=production
    export AEGIS_DEBUG=false
    
    # Rolling update strategy
    log_info "Performing rolling update..."
    
    # Scale up new instances
    docker-compose up -d --scale aegis-core=2
    
    # Wait for new instances to be healthy
    sleep 30
    wait_for_health_check "http://production.aegis.local:8080/health"
    
    # Scale down old instances
    docker-compose up -d --scale aegis-core=1
    
    log_success "Production deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_header "Deploying to Kubernetes..."
    
    local k8s_dir="$AEGIS_DIR/k8s"
    
    if [[ ! -d "$k8s_dir" ]]; then
        log_error "Kubernetes manifests directory not found: $k8s_dir"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace aegis --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ConfigMaps and Secrets
    if [[ -f "$k8s_dir/configmap.yaml" ]]; then
        kubectl apply -f "$k8s_dir/configmap.yaml" -n aegis
    fi
    
    if [[ -f "$k8s_dir/secrets.yaml" ]]; then
        kubectl apply -f "$k8s_dir/secrets.yaml" -n aegis
    fi
    
    # Deploy using Helm if chart exists
    if [[ -d "$k8s_dir/helm/aegis" ]]; then
        log_info "Deploying using Helm..."
        helm upgrade --install aegis "$k8s_dir/helm/aegis" \
            --namespace aegis \
            --set image.tag="$VERSION" \
            --set environment="$ENVIRONMENT"
    else
        # Deploy using kubectl
        log_info "Deploying using kubectl..."
        kubectl apply -f "$k8s_dir/" -n aegis
    fi
    
    # Wait for rollout to complete
    kubectl rollout status deployment/aegis-core -n aegis --timeout=600s
    
    log_success "Kubernetes deployment completed"
}

# Wait for health check
wait_for_health_check() {
    local url="$1"
    local timeout="${2:-$HEALTH_CHECK_TIMEOUT}"
    local interval=10
    local elapsed=0
    
    log_info "Waiting for health check at $url..."
    
    while [[ $elapsed -lt $timeout ]]; do
        if curl -f -s "$url" >/dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        log_info "Health check failed, retrying in ${interval}s... (${elapsed}/${timeout}s)"
    done
    
    log_error "Health check timeout after ${timeout}s"
    return 1
}

# Rollback deployment
rollback_deployment() {
    log_header "Rolling back deployment..."
    
    local backup_dir
    backup_dir=$(find "$AEGIS_DIR/backups" -type d -name "*" | sort -r | head -1)
    
    if [[ -z "$backup_dir" ]]; then
        log_error "No backup found for rollback"
        exit 1
    fi
    
    log_info "Rolling back to backup: $backup_dir"
    
    # Restore configuration
    cp "$backup_dir/.env" "$AEGIS_DIR/"
    cp -r "$backup_dir/config/"* "$AEGIS_DIR/config/"
    
    # Restore database
    if [[ -f "$backup_dir/postgres_backup.sql" ]]; then
        log_info "Restoring PostgreSQL database..."
        PGPASSWORD="$POSTGRES_PASSWORD" psql \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            < "$backup_dir/postgres_backup.sql"
    fi
    
    # Restore Redis data
    if [[ -f "$backup_dir/redis_backup.rdb" ]]; then
        log_info "Restoring Redis data..."
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --rdb "$backup_dir/redis_backup.rdb"
    fi
    
    # Restart services
    case "$ENVIRONMENT" in
        local)
            deploy_local
            ;;
        staging)
            deploy_staging
            ;;
        production)
            deploy_production
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
    esac
    
    log_success "Rollback completed"
}

# Post-deployment tasks
post_deployment() {
    log_header "Running post-deployment tasks..."
    
    # Run database migrations
    if [[ -f "$AEGIS_DIR/migrations/migrate.py" ]]; then
        log_info "Running database migrations..."
        python "$AEGIS_DIR/migrations/migrate.py"
    fi
    
    # Clear caches
    if [[ -n "${REDIS_HOST:-}" ]]; then
        log_info "Clearing Redis cache..."
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" FLUSHALL
    fi
    
    # Send deployment notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        log_info "Sending deployment notification..."
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"AEGIS deployment completed: $ENVIRONMENT v$VERSION\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
    
    log_success "Post-deployment tasks completed"
}

# Cleanup old images and containers
cleanup() {
    log_header "Cleaning up old images and containers..."
    
    # Remove old images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove old backups (keep last 10)
    find "$AEGIS_DIR/backups" -type d -name "*" | sort -r | tail -n +11 | xargs rm -rf
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_header "AEGIS Framework Deployment"
    echo "Advanced Encrypted Governance and Intelligence System"
    echo "======================================================="
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "Timestamp: $TIMESTAMP"
    echo ""
    
    # Parse arguments
    parse_arguments "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Handle rollback
    if [[ "$ROLLBACK" == true ]]; then
        rollback_deployment
        exit 0
    fi
    
    # Run tests
    run_tests
    
    # Create backup
    create_backup
    
    # Build images
    build_images
    
    # Push images if requested
    push_images
    
    # Exit if build-only
    if [[ "$BUILD_ONLY" == true ]]; then
        log_success "Build completed, exiting as requested"
        exit 0
    fi
    
    # Deploy based on environment
    case "$ENVIRONMENT" in
        local)
            deploy_local
            ;;
        staging)
            deploy_staging
            ;;
        production)
            deploy_production
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    # Post-deployment tasks
    post_deployment
    
    # Cleanup
    cleanup
    
    log_success "Deployment completed successfully!"
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "Health Check: $(case "$ENVIRONMENT" in
        local) echo "http://localhost:8080/health" ;;
        staging) echo "http://staging.aegis.local:8080/health" ;;
        production) echo "http://production.aegis.local:8080/health" ;;
        kubernetes) echo "kubectl get pods -n aegis" ;;
    esac)"
}

# Run main function with all arguments
main "$@"