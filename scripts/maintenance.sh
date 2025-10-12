#!/bin/bash

# AEGIS Framework - Maintenance Script
# Automated system maintenance and optimization

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/maintenance.log"
BACKUP_DIR="${PROJECT_ROOT}/backups"
TEMP_DIR="/tmp/aegis_maintenance"
MAX_LOG_SIZE="100M"
MAX_BACKUP_AGE=30
VERBOSE=false
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Maintenance statistics
CLEANED_FILES=0
FREED_SPACE=0
ROTATED_LOGS=0
OPTIMIZED_DATABASES=0

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    if [[ "$VERBOSE" == "true" ]]; then
        case "$level" in
            "ERROR") echo -e "${RED}[ERROR]${NC} $message" >&2 ;;
            "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
            "INFO")  echo -e "${BLUE}[INFO]${NC} $message" ;;
            "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        esac
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get file size in bytes
get_file_size() {
    local file="$1"
    if [[ -f "$file" ]]; then
        stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Convert bytes to human readable format
bytes_to_human() {
    local bytes="$1"
    if [[ "$bytes" -lt 1024 ]]; then
        echo "${bytes}B"
    elif [[ "$bytes" -lt 1048576 ]]; then
        echo "$((bytes / 1024))KB"
    elif [[ "$bytes" -lt 1073741824 ]]; then
        echo "$((bytes / 1048576))MB"
    else
        echo "$((bytes / 1073741824))GB"
    fi
}

# Clean temporary files
clean_temp_files() {
    log "INFO" "Cleaning temporary files..."
    
    local temp_dirs=(
        "/tmp"
        "/var/tmp"
        "${PROJECT_ROOT}/temp"
        "${PROJECT_ROOT}/.cache"
        "${PROJECT_ROOT}/node_modules/.cache"
        "${PROJECT_ROOT}/__pycache__"
    )
    
    local cleaned_size=0
    
    for temp_dir in "${temp_dirs[@]}"; do
        if [[ -d "$temp_dir" ]]; then
            # Find and clean AEGIS-related temp files
            while IFS= read -r -d '' file; do
                if [[ "$DRY_RUN" == "true" ]]; then
                    log "INFO" "Would delete: $file"
                else
                    local file_size
                    file_size=$(get_file_size "$file")
                    rm -f "$file" 2>/dev/null || true
                    cleaned_size=$((cleaned_size + file_size))
                    ((CLEANED_FILES++))
                fi
            done < <(find "$temp_dir" -name "*aegis*" -type f -mtime +1 -print0 2>/dev/null || true)
        fi
    done
    
    # Clean Python cache files
    if [[ -d "$PROJECT_ROOT" ]]; then
        while IFS= read -r -d '' file; do
            if [[ "$DRY_RUN" == "true" ]]; then
                log "INFO" "Would delete: $file"
            else
                local file_size
                file_size=$(get_file_size "$file")
                rm -f "$file" 2>/dev/null || true
                cleaned_size=$((cleaned_size + file_size))
                ((CLEANED_FILES++))
            fi
        done < <(find "$PROJECT_ROOT" -name "*.pyc" -o -name "*.pyo" -o -name "__pycache__" -type f -print0 2>/dev/null || true)
    fi
    
    FREED_SPACE=$((FREED_SPACE + cleaned_size))
    log "SUCCESS" "Cleaned $CLEANED_FILES temporary files, freed $(bytes_to_human $cleaned_size)"
}

# Rotate log files
rotate_logs() {
    log "INFO" "Rotating log files..."
    
    local log_dir="${PROJECT_ROOT}/logs"
    
    if [[ ! -d "$log_dir" ]]; then
        log "WARN" "Log directory does not exist: $log_dir"
        return 0
    fi
    
    # Find large log files
    while IFS= read -r -d '' logfile; do
        local file_size
        file_size=$(get_file_size "$logfile")
        local max_size_bytes=104857600  # 100MB
        
        if [[ "$file_size" -gt "$max_size_bytes" ]]; then
            local basename
            basename=$(basename "$logfile")
            local timestamp
            timestamp=$(date '+%Y%m%d_%H%M%S')
            local rotated_name="${basename}.${timestamp}"
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log "INFO" "Would rotate: $logfile -> $rotated_name"
            else
                mv "$logfile" "${log_dir}/${rotated_name}"
                touch "$logfile"
                chmod 644 "$logfile"
                
                # Compress rotated log
                if command_exists gzip; then
                    gzip "${log_dir}/${rotated_name}"
                fi
                
                ((ROTATED_LOGS++))
                log "SUCCESS" "Rotated log file: $basename"
            fi
        fi
    done < <(find "$log_dir" -name "*.log" -type f -print0 2>/dev/null || true)
    
    # Clean old rotated logs
    if [[ "$DRY_RUN" == "false" ]]; then
        find "$log_dir" -name "*.log.*" -type f -mtime +$MAX_BACKUP_AGE -delete 2>/dev/null || true
    fi
    
    log "SUCCESS" "Rotated $ROTATED_LOGS log files"
}

# Clean old backups
clean_old_backups() {
    log "INFO" "Cleaning old backups..."
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log "INFO" "Backup directory does not exist: $BACKUP_DIR"
        return 0
    fi
    
    local cleaned_backups=0
    local cleaned_size=0
    
    # Clean backups older than MAX_BACKUP_AGE days
    while IFS= read -r -d '' backup; do
        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "Would delete old backup: $backup"
        else
            local file_size
            file_size=$(get_file_size "$backup")
            rm -f "$backup"
            cleaned_size=$((cleaned_size + file_size))
            ((cleaned_backups++))
        fi
    done < <(find "$BACKUP_DIR" -type f -mtime +$MAX_BACKUP_AGE -print0 2>/dev/null || true)
    
    FREED_SPACE=$((FREED_SPACE + cleaned_size))
    log "SUCCESS" "Cleaned $cleaned_backups old backups, freed $(bytes_to_human $cleaned_size)"
}

# Optimize databases
optimize_databases() {
    log "INFO" "Optimizing databases..."
    
    # Check if Docker is available
    if ! command_exists docker; then
        log "WARN" "Docker not available, skipping database optimization"
        return 0
    fi
    
    # Optimize PostgreSQL
    if docker ps | grep -q postgres; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "Would optimize PostgreSQL database"
        else
            log "INFO" "Running PostgreSQL VACUUM and ANALYZE..."
            docker exec -i $(docker ps | grep postgres | awk '{print $1}') psql -U postgres -d aegis -c "VACUUM ANALYZE;" 2>/dev/null || true
            ((OPTIMIZED_DATABASES++))
            log "SUCCESS" "Optimized PostgreSQL database"
        fi
    fi
    
    # Optimize Redis
    if docker ps | grep -q redis; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "Would optimize Redis database"
        else
            log "INFO" "Running Redis BGREWRITEAOF..."
            docker exec -i $(docker ps | grep redis | awk '{print $1}') redis-cli BGREWRITEAOF 2>/dev/null || true
            ((OPTIMIZED_DATABASES++))
            log "SUCCESS" "Optimized Redis database"
        fi
    fi
    
    log "SUCCESS" "Optimized $OPTIMIZED_DATABASES databases"
}

# Clean Docker resources
clean_docker() {
    log "INFO" "Cleaning Docker resources..."
    
    if ! command_exists docker; then
        log "WARN" "Docker not available, skipping Docker cleanup"
        return 0
    fi
    
    local cleaned_size=0
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would clean Docker resources"
        docker system df 2>/dev/null || true
    else
        # Remove unused containers, networks, images, and build cache
        local before_size
        before_size=$(docker system df --format "table {{.Size}}" 2>/dev/null | tail -n +2 | head -1 || echo "0B")
        
        docker system prune -f --volumes 2>/dev/null || true
        
        local after_size
        after_size=$(docker system df --format "table {{.Size}}" 2>/dev/null | tail -n +2 | head -1 || echo "0B")
        
        log "SUCCESS" "Cleaned Docker resources"
    fi
}

# Update system packages
update_packages() {
    log "INFO" "Updating system packages..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Would update system packages"
        return 0
    fi
    
    # Update based on the system
    if command_exists apt-get; then
        # Debian/Ubuntu
        apt-get update -qq 2>/dev/null || true
        apt-get upgrade -y -qq 2>/dev/null || true
        apt-get autoremove -y -qq 2>/dev/null || true
        log "SUCCESS" "Updated packages using apt-get"
    elif command_exists yum; then
        # RHEL/CentOS
        yum update -y -q 2>/dev/null || true
        yum autoremove -y -q 2>/dev/null || true
        log "SUCCESS" "Updated packages using yum"
    elif command_exists dnf; then
        # Fedora
        dnf update -y -q 2>/dev/null || true
        dnf autoremove -y -q 2>/dev/null || true
        log "SUCCESS" "Updated packages using dnf"
    elif command_exists pacman; then
        # Arch Linux
        pacman -Syu --noconfirm -q 2>/dev/null || true
        pacman -Rns $(pacman -Qtdq) --noconfirm -q 2>/dev/null || true
        log "SUCCESS" "Updated packages using pacman"
    else
        log "WARN" "No supported package manager found"
    fi
}

# Check and repair file permissions
fix_permissions() {
    log "INFO" "Checking and fixing file permissions..."
    
    local fixed_files=0
    
    # Fix script permissions
    while IFS= read -r -d '' script; do
        if [[ ! -x "$script" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                log "INFO" "Would fix permissions for: $script"
            else
                chmod +x "$script"
                ((fixed_files++))
                log "INFO" "Fixed permissions for: $script"
            fi
        fi
    done < <(find "${PROJECT_ROOT}/scripts" -name "*.sh" -type f -print0 2>/dev/null || true)
    
    # Fix sensitive file permissions
    local sensitive_files=(
        "${PROJECT_ROOT}/.env"
        "${PROJECT_ROOT}/config/ssl"
        "${PROJECT_ROOT}/config/keys"
    )
    
    for file in "${sensitive_files[@]}"; do
        if [[ -e "$file" ]]; then
            local current_perms
            current_perms=$(stat -c "%a" "$file" 2>/dev/null || stat -f "%A" "$file" 2>/dev/null || echo "unknown")
            
            if [[ "$current_perms" != "600" && "$current_perms" != "700" ]]; then
                if [[ "$DRY_RUN" == "true" ]]; then
                    log "INFO" "Would fix permissions for: $file"
                else
                    if [[ -d "$file" ]]; then
                        chmod 700 "$file"
                    else
                        chmod 600 "$file"
                    fi
                    ((fixed_files++))
                    log "INFO" "Fixed permissions for: $file"
                fi
            fi
        fi
    done
    
    log "SUCCESS" "Fixed permissions for $fixed_files files"
}

# Generate maintenance report
generate_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local freed_space_human
    freed_space_human=$(bytes_to_human $FREED_SPACE)
    
    echo
    echo "========================================="
    echo "AEGIS Framework Maintenance Report"
    echo "========================================="
    echo "Timestamp: $timestamp"
    echo "Mode: $([ "$DRY_RUN" == "true" ] && echo "DRY RUN" || echo "LIVE")"
    echo
    echo "Summary:"
    echo "  - Cleaned files: $CLEANED_FILES"
    echo "  - Freed space: $freed_space_human"
    echo "  - Rotated logs: $ROTATED_LOGS"
    echo "  - Optimized databases: $OPTIMIZED_DATABASES"
    echo
    echo "Maintenance completed successfully!"
    echo "========================================="
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

AEGIS Framework Maintenance Script

OPTIONS:
    -v, --verbose       Enable verbose output
    -n, --dry-run       Show what would be done without making changes
    -q, --quick         Run quick maintenance (skip package updates)
    -f, --full          Run full maintenance including system updates
    -h, --help          Show this help message

MAINTENANCE TASKS:
    - Clean temporary files and caches
    - Rotate large log files
    - Clean old backups
    - Optimize databases (PostgreSQL, Redis)
    - Clean Docker resources
    - Update system packages (with --full)
    - Fix file permissions

EXAMPLES:
    $0                  Run standard maintenance
    $0 -v               Run with verbose output
    $0 -n               Dry run (show what would be done)
    $0 -f               Run full maintenance including updates
    $0 -q               Run quick maintenance only

SCHEDULING:
    Add to crontab for automated maintenance:
    0 2 * * 0 /path/to/maintenance.sh -q >> /var/log/aegis_maintenance.log 2>&1
EOF
}

# Main function
main() {
    local quick_mode=false
    local full_mode=false
    
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
            -q|--quick)
                quick_mode=true
                shift
                ;;
            -f|--full)
                full_mode=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Create necessary directories
    mkdir -p "$(dirname "$LOG_FILE")" "$TEMP_DIR"
    
    # Start maintenance
    log "INFO" "Starting AEGIS Framework maintenance..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Running in DRY RUN mode - no changes will be made"
    fi
    
    # Run maintenance tasks
    clean_temp_files
    rotate_logs
    clean_old_backups
    optimize_databases
    clean_docker
    fix_permissions
    
    # Run additional tasks based on mode
    if [[ "$full_mode" == "true" && "$quick_mode" == "false" ]]; then
        update_packages
    fi
    
    # Generate report
    generate_report
    
    # Cleanup
    rm -rf "$TEMP_DIR" 2>/dev/null || true
    
    log "SUCCESS" "Maintenance completed successfully"
}

# Run main function
main "$@"