#!/bin/bash
# AEGIS Framework - Update Script
# Automated system updates and dependency management

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/update.log"
BACKUP_DIR="$PROJECT_ROOT/backups/pre-update"
TEMP_DIR="/tmp/aegis_update_$$"

# Default values
VERBOSE=false
DRY_RUN=false
FORCE=false
SKIP_BACKUP=false
SKIP_TESTS=false
UPDATE_SYSTEM=false
UPDATE_DOCKER=false
UPDATE_PYTHON=false
UPDATE_NODE=false
UPDATE_ALL=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Update statistics
UPDATED_PACKAGES=0
UPDATED_SERVICES=0
FAILED_UPDATES=0

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Ensure log directory exists
    mkdir -p "$(dirname "$LOG_FILE")"
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    if [[ "$VERBOSE" == "true" ]]; then
        case "$level" in
            "ERROR")
                echo -e "${RED}[ERROR]${NC} $message" >&2
                ;;
            "WARN")
                echo -e "${YELLOW}[WARN]${NC} $message"
                ;;
            "INFO")
                echo -e "${BLUE}[INFO]${NC} $message"
                ;;
            "SUCCESS")
                echo -e "${GREEN}[SUCCESS]${NC} $message"
                ;;
        esac
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    log "INFO" "Checking system requirements..."
    
    local missing_deps=()
    
    # Check essential commands
    for cmd in git curl wget; do
        if ! command_exists "$cmd"; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "ERROR" "Missing required dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    # Check disk space (require at least 1GB free)
    local available_space
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 1048576 ]]; then
        log "WARN" "Low disk space detected. Available: $(($available_space / 1024))MB"
    fi
    
    log "SUCCESS" "System requirements check passed"
}

# Create pre-update backup
create_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log "INFO" "Skipping backup as requested"
        return
    fi
    
    log "INFO" "Creating pre-update backup..."
    
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_path="$BACKUP_DIR/backup_$backup_timestamp"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would create backup at: $backup_path"
        return
    fi
    
    mkdir -p "$backup_path"
    
    # Backup configuration files
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        cp -r "$PROJECT_ROOT/config" "$backup_path/"
    fi
    
    # Backup environment files
    for env_file in .env .env.local .env.production; do
        if [[ -f "$PROJECT_ROOT/$env_file" ]]; then
            cp "$PROJECT_ROOT/$env_file" "$backup_path/"
        fi
    done
    
    # Backup Docker compose files
    for compose_file in docker-compose.yml docker-compose.override.yml; do
        if [[ -f "$PROJECT_ROOT/$compose_file" ]]; then
            cp "$PROJECT_ROOT/$compose_file" "$backup_path/"
        fi
    done
    
    # Backup package files
    for pkg_file in package.json requirements.txt Pipfile pyproject.toml; do
        if [[ -f "$PROJECT_ROOT/$pkg_file" ]]; then
            cp "$PROJECT_ROOT/$pkg_file" "$backup_path/"
        fi
    done
    
    log "SUCCESS" "Backup created at: $backup_path"
}

# Update system packages
update_system_packages() {
    if [[ "$UPDATE_SYSTEM" != "true" && "$UPDATE_ALL" != "true" ]]; then
        return
    fi
    
    log "INFO" "Updating system packages..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would update system packages"
        return
    fi
    
    # Detect package manager and update
    if command_exists apt-get; then
        log "INFO" "Updating packages using apt..."
        sudo apt-get update -qq
        sudo apt-get upgrade -y -qq
        sudo apt-get autoremove -y -qq
        ((UPDATED_PACKAGES++))
    elif command_exists yum; then
        log "INFO" "Updating packages using yum..."
        sudo yum update -y -q
        ((UPDATED_PACKAGES++))
    elif command_exists dnf; then
        log "INFO" "Updating packages using dnf..."
        sudo dnf update -y -q
        ((UPDATED_PACKAGES++))
    elif command_exists pacman; then
        log "INFO" "Updating packages using pacman..."
        sudo pacman -Syu --noconfirm --quiet
        ((UPDATED_PACKAGES++))
    elif command_exists brew; then
        log "INFO" "Updating packages using Homebrew..."
        brew update -q
        brew upgrade -q
        brew cleanup -q
        ((UPDATED_PACKAGES++))
    else
        log "WARN" "No supported package manager found"
    fi
    
    log "SUCCESS" "System packages updated"
}

# Update Docker and Docker Compose
update_docker() {
    if [[ "$UPDATE_DOCKER" != "true" && "$UPDATE_ALL" != "true" ]]; then
        return
    fi
    
    log "INFO" "Updating Docker components..."
    
    if ! command_exists docker; then
        log "WARN" "Docker not installed, skipping Docker update"
        return
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would update Docker components"
        return
    fi
    
    # Update Docker images
    log "INFO" "Pulling latest Docker images..."
    
    # Get list of images used in docker-compose
    if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        local images
        images=$(grep -E "^\s*image:" "$PROJECT_ROOT/docker-compose.yml" | awk '{print $2}' | sort -u)
        
        for image in $images; do
            log "INFO" "Pulling image: $image"
            if docker pull "$image" >/dev/null 2>&1; then
                log "SUCCESS" "Updated image: $image"
                ((UPDATED_SERVICES++))
            else
                log "ERROR" "Failed to update image: $image"
                ((FAILED_UPDATES++))
            fi
        done
    fi
    
    # Update Docker Compose if installed via pip
    if command_exists pip3 && pip3 list | grep -q docker-compose; then
        log "INFO" "Updating Docker Compose..."
        pip3 install --upgrade docker-compose >/dev/null 2>&1
        ((UPDATED_PACKAGES++))
    fi
    
    log "SUCCESS" "Docker components updated"
}

# Update Python dependencies
update_python_deps() {
    if [[ "$UPDATE_PYTHON" != "true" && "$UPDATE_ALL" != "true" ]]; then
        return
    fi
    
    log "INFO" "Updating Python dependencies..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would update Python dependencies"
        return
    fi
    
    cd "$PROJECT_ROOT"
    
    # Update pip itself
    if command_exists pip3; then
        log "INFO" "Updating pip..."
        pip3 install --upgrade pip >/dev/null 2>&1
    fi
    
    # Update requirements.txt dependencies
    if [[ -f "requirements.txt" ]]; then
        log "INFO" "Updating requirements.txt dependencies..."
        if pip3 install --upgrade -r requirements.txt >/dev/null 2>&1; then
            log "SUCCESS" "Updated requirements.txt dependencies"
            ((UPDATED_PACKAGES++))
        else
            log "ERROR" "Failed to update requirements.txt dependencies"
            ((FAILED_UPDATES++))
        fi
    fi
    
    # Update Pipfile dependencies
    if [[ -f "Pipfile" ]] && command_exists pipenv; then
        log "INFO" "Updating Pipfile dependencies..."
        if pipenv update >/dev/null 2>&1; then
            log "SUCCESS" "Updated Pipfile dependencies"
            ((UPDATED_PACKAGES++))
        else
            log "ERROR" "Failed to update Pipfile dependencies"
            ((FAILED_UPDATES++))
        fi
    fi
    
    # Update pyproject.toml dependencies
    if [[ -f "pyproject.toml" ]] && command_exists poetry; then
        log "INFO" "Updating pyproject.toml dependencies..."
        if poetry update >/dev/null 2>&1; then
            log "SUCCESS" "Updated pyproject.toml dependencies"
            ((UPDATED_PACKAGES++))
        else
            log "ERROR" "Failed to update pyproject.toml dependencies"
            ((FAILED_UPDATES++))
        fi
    fi
    
    log "SUCCESS" "Python dependencies updated"
}

# Update Node.js dependencies
update_node_deps() {
    if [[ "$UPDATE_NODE" != "true" && "$UPDATE_ALL" != "true" ]]; then
        return
    fi
    
    log "INFO" "Updating Node.js dependencies..."
    
    if ! command_exists npm; then
        log "WARN" "Node.js/npm not installed, skipping Node.js update"
        return
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would update Node.js dependencies"
        return
    fi
    
    cd "$PROJECT_ROOT"
    
    # Update npm itself
    log "INFO" "Updating npm..."
    npm install -g npm@latest >/dev/null 2>&1
    
    # Update package.json dependencies
    if [[ -f "package.json" ]]; then
        log "INFO" "Updating package.json dependencies..."
        if npm update >/dev/null 2>&1; then
            log "SUCCESS" "Updated package.json dependencies"
            ((UPDATED_PACKAGES++))
        else
            log "ERROR" "Failed to update package.json dependencies"
            ((FAILED_UPDATES++))
        fi
        
        # Audit and fix vulnerabilities
        log "INFO" "Auditing and fixing npm vulnerabilities..."
        npm audit fix >/dev/null 2>&1 || true
    fi
    
    # Update yarn dependencies if yarn.lock exists
    if [[ -f "yarn.lock" ]] && command_exists yarn; then
        log "INFO" "Updating yarn dependencies..."
        if yarn upgrade >/dev/null 2>&1; then
            log "SUCCESS" "Updated yarn dependencies"
            ((UPDATED_PACKAGES++))
        else
            log "ERROR" "Failed to update yarn dependencies"
            ((FAILED_UPDATES++))
        fi
    fi
    
    log "SUCCESS" "Node.js dependencies updated"
}

# Update AEGIS codebase
update_aegis_code() {
    log "INFO" "Updating AEGIS codebase..."
    
    cd "$PROJECT_ROOT"
    
    # Check if we're in a git repository
    if [[ ! -d ".git" ]]; then
        log "WARN" "Not a git repository, skipping code update"
        return
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would update AEGIS codebase from git"
        return
    fi
    
    # Stash any local changes
    local stash_created=false
    if ! git diff-index --quiet HEAD --; then
        log "INFO" "Stashing local changes..."
        git stash push -m "Pre-update stash $(date '+%Y-%m-%d %H:%M:%S')"
        stash_created=true
    fi
    
    # Fetch latest changes
    log "INFO" "Fetching latest changes..."
    git fetch origin
    
    # Get current branch
    local current_branch
    current_branch=$(git branch --show-current)
    
    # Pull latest changes
    log "INFO" "Pulling latest changes for branch: $current_branch"
    if git pull origin "$current_branch"; then
        log "SUCCESS" "Updated AEGIS codebase"
        ((UPDATED_SERVICES++))
    else
        log "ERROR" "Failed to update AEGIS codebase"
        ((FAILED_UPDATES++))
        
        # Restore stashed changes if any
        if [[ "$stash_created" == "true" ]]; then
            git stash pop
        fi
        return 1
    fi
    
    # Restore stashed changes if any
    if [[ "$stash_created" == "true" ]]; then
        log "INFO" "Restoring stashed changes..."
        git stash pop || log "WARN" "Could not restore stashed changes automatically"
    fi
    
    log "SUCCESS" "AEGIS codebase updated"
}

# Run post-update tasks
run_post_update_tasks() {
    log "INFO" "Running post-update tasks..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would run post-update tasks"
        return
    fi
    
    cd "$PROJECT_ROOT"
    
    # Rebuild Docker containers if docker-compose.yml exists
    if [[ -f "docker-compose.yml" ]] && command_exists docker-compose; then
        log "INFO" "Rebuilding Docker containers..."
        docker-compose build --no-cache >/dev/null 2>&1 || log "WARN" "Failed to rebuild containers"
    fi
    
    # Install/update pre-commit hooks if .pre-commit-config.yaml exists
    if [[ -f ".pre-commit-config.yaml" ]] && command_exists pre-commit; then
        log "INFO" "Updating pre-commit hooks..."
        pre-commit install >/dev/null 2>&1
        pre-commit autoupdate >/dev/null 2>&1
    fi
    
    # Clear Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Clear Node.js cache if node_modules exists
    if [[ -d "node_modules" ]] && command_exists npm; then
        npm cache clean --force >/dev/null 2>&1 || true
    fi
    
    log "SUCCESS" "Post-update tasks completed"
}

# Run tests after update
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log "INFO" "Skipping tests as requested"
        return
    fi
    
    log "INFO" "Running tests after update..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would run tests"
        return
    fi
    
    cd "$PROJECT_ROOT"
    
    # Run Python tests
    if [[ -f "pytest.ini" ]] || [[ -d "tests" ]]; then
        log "INFO" "Running Python tests..."
        if command_exists pytest; then
            pytest --tb=short -q >/dev/null 2>&1 || log "WARN" "Some Python tests failed"
        fi
    fi
    
    # Run Node.js tests
    if [[ -f "package.json" ]] && command_exists npm; then
        if npm run test --if-present >/dev/null 2>&1; then
            log "SUCCESS" "Node.js tests passed"
        else
            log "WARN" "Some Node.js tests failed"
        fi
    fi
    
    # Run linting
    if command_exists flake8; then
        flake8 . >/dev/null 2>&1 || log "WARN" "Linting issues found"
    fi
    
    log "SUCCESS" "Tests completed"
}

# Generate update report
generate_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo ""
    echo "========================================="
    echo "AEGIS Framework Update Report"
    echo "========================================="
    echo "Timestamp: $timestamp"
    echo "Mode: $(if [[ "$DRY_RUN" == "true" ]]; then echo "DRY RUN"; else echo "LIVE"; fi)"
    echo ""
    echo "Summary:"
    echo "  - Updated packages: $UPDATED_PACKAGES"
    echo "  - Updated services: $UPDATED_SERVICES"
    echo "  - Failed updates: $FAILED_UPDATES"
    echo ""
    
    if [[ $FAILED_UPDATES -gt 0 ]]; then
        echo -e "${YELLOW}Update completed with some failures!${NC}"
        echo "Check the log file for details: $LOG_FILE"
    else
        echo -e "${GREEN}Update completed successfully!${NC}"
    fi
    
    echo "========================================="
}

# Cleanup function
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

AEGIS Framework Update Script

OPTIONS:
    -v, --verbose       Enable verbose output
    -n, --dry-run       Show what would be done without making changes
    -f, --force         Force update even if checks fail
    --skip-backup       Skip creating pre-update backup
    --skip-tests        Skip running tests after update
    --system            Update system packages
    --docker            Update Docker components
    --python            Update Python dependencies
    --node              Update Node.js dependencies
    --all               Update everything (system, docker, python, node)
    -h, --help          Show this help message

UPDATE COMPONENTS:
    - AEGIS codebase (git pull)
    - System packages (apt, yum, brew, etc.)
    - Docker images and containers
    - Python dependencies (pip, pipenv, poetry)
    - Node.js dependencies (npm, yarn)
    - Post-update tasks (rebuild, cache clear)

EXAMPLES:
    $0                  Update AEGIS codebase only
    $0 --all            Update everything
    $0 --python --node  Update Python and Node.js dependencies
    $0 --dry-run --all  Dry run of full update
    $0 --verbose --system  Update system packages with verbose output

SCHEDULING:
    Add to crontab for automated updates:
    0 2 * * 0 /path/to/update.sh --python --node >> /var/log/aegis-update.log 2>&1
EOF
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --system)
                UPDATE_SYSTEM=true
                shift
                ;;
            --docker)
                UPDATE_DOCKER=true
                shift
                ;;
            --python)
                UPDATE_PYTHON=true
                shift
                ;;
            --node)
                UPDATE_NODE=true
                shift
                ;;
            --all)
                UPDATE_ALL=true
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
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    trap cleanup EXIT
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Start update process
    log "INFO" "Starting AEGIS Framework update..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Running in DRY RUN mode - no changes will be made"
    fi
    
    # Run update steps
    check_requirements
    create_backup
    update_aegis_code
    update_system_packages
    update_docker
    update_python_deps
    update_node_deps
    run_post_update_tasks
    run_tests
    
    # Generate report
    generate_report
    
    log "SUCCESS" "Update process completed"
    
    # Exit with error code if there were failures
    if [[ $FAILED_UPDATES -gt 0 ]]; then
        exit 1
    fi
}

# Run main function with all arguments
main "$@"