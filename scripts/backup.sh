#!/bin/bash

# AEGIS Framework - Backup Script
# Advanced Encrypted Governance and Intelligence System
# This script provides comprehensive backup and restore functionality

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
BACKUP_DIR="${BACKUP_DIR:-$AEGIS_DIR/backups}"
LOG_FILE="$AEGIS_DIR/logs/backup.log"
RETENTION_DAYS=30
COMPRESSION_LEVEL=6
ENCRYPTION_ENABLED=true
BACKUP_PREFIX="aegis_backup"
MAX_PARALLEL_JOBS=4

# Default values
BACKUP_TYPE="full"
RESTORE_MODE=false
LIST_BACKUPS=false
CLEANUP_MODE=false
VERIFY_MODE=false
VERBOSE=false
DRY_RUN=false
REMOTE_BACKUP=false
INCREMENTAL=false

# Logging functions
log_info() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[INFO]${NC} $message"
    echo "[$timestamp] [INFO] $message" >> "$LOG_FILE"
}

log_success() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}[SUCCESS]${NC} $message"
    echo "[$timestamp] [SUCCESS] $message" >> "$LOG_FILE"
}

log_warning() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[WARNING]${NC} $message"
    echo "[$timestamp] [WARNING] $message" >> "$LOG_FILE"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${RED}[ERROR]${NC} $message"
    echo "[$timestamp] [ERROR] $message" >> "$LOG_FILE"
}

log_header() {
    local message="$1"
    echo -e "${PURPLE}[AEGIS BACKUP]${NC} $message"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Show help
show_help() {
    cat << EOF
AEGIS Framework Backup Script

Usage: $0 [OPTIONS] [COMMAND]

COMMANDS:
    backup              Create a backup (default)
    restore             Restore from backup
    list                List available backups
    cleanup             Clean up old backups
    verify              Verify backup integrity

OPTIONS:
    -t, --type TYPE     Backup type: full, database, config, logs (default: full)
    -d, --dir DIR       Backup directory (default: ./backups)
    -r, --retention N   Retention period in days (default: 30)
    -c, --compression N Compression level 1-9 (default: 6)
    -e, --encrypt       Enable encryption (default: true)
    -i, --incremental   Create incremental backup
    --remote            Enable remote backup
    --dry-run           Show what would be done without executing
    -v, --verbose       Enable verbose output
    -h, --help          Show this help message

BACKUP TYPES:
    full        - Complete system backup (databases, config, logs, certs)
    database    - PostgreSQL and Redis databases only
    config      - Configuration files and certificates
    logs        - Log files and monitoring data
    docker      - Docker volumes and container data

EXAMPLES:
    $0                          # Full backup
    $0 backup --type database   # Database backup only
    $0 restore --file backup.tar.gz.enc  # Restore from specific backup
    $0 list                     # List all backups
    $0 cleanup --retention 7    # Clean backups older than 7 days
    $0 verify --file backup.tar.gz.enc   # Verify backup integrity

ENVIRONMENT VARIABLES:
    BACKUP_DIR          - Backup directory path
    BACKUP_ENCRYPTION_KEY - Encryption key for backups
    REMOTE_BACKUP_URL   - Remote backup destination (rsync, s3, etc.)
    POSTGRES_PASSWORD   - PostgreSQL password for database backups
    REDIS_PASSWORD      - Redis password for database backups

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            backup)
                # Default action
                shift
                ;;
            restore)
                RESTORE_MODE=true
                shift
                ;;
            list)
                LIST_BACKUPS=true
                shift
                ;;
            cleanup)
                CLEANUP_MODE=true
                shift
                ;;
            verify)
                VERIFY_MODE=true
                shift
                ;;
            -t|--type)
                BACKUP_TYPE="$2"
                shift 2
                ;;
            -d|--dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -c|--compression)
                COMPRESSION_LEVEL="$2"
                shift 2
                ;;
            -e|--encrypt)
                ENCRYPTION_ENABLED=true
                shift
                ;;
            --no-encrypt)
                ENCRYPTION_ENABLED=false
                shift
                ;;
            -i|--incremental)
                INCREMENTAL=true
                shift
                ;;
            --remote)
                REMOTE_BACKUP=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --file)
                BACKUP_FILE="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Initialize backup system
initialize_backup() {
    log_header "Initializing AEGIS backup system..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Load environment variables
    if [[ -f "$AEGIS_DIR/.env" ]]; then
        set -a
        source "$AEGIS_DIR/.env"
        set +a
    fi
    
    # Check required tools
    local required_tools=("tar" "gzip")
    for tool in "${required_tools[@]}"; do
        if ! command_exists "$tool"; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check optional tools
    if [[ "$ENCRYPTION_ENABLED" == true ]] && ! command_exists "openssl"; then
        log_warning "OpenSSL not found, encryption will be disabled"
        ENCRYPTION_ENABLED=false
    fi
    
    if [[ "$REMOTE_BACKUP" == true ]] && ! command_exists "rsync"; then
        log_warning "rsync not found, remote backup will be disabled"
        REMOTE_BACKUP=false
    fi
    
    log_success "Backup system initialized"
}

# Generate backup filename
generate_backup_filename() {
    local backup_type="$1"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local hostname=$(hostname -s)
    
    if [[ "$INCREMENTAL" == true ]]; then
        echo "${BACKUP_PREFIX}_${backup_type}_incremental_${hostname}_${timestamp}"
    else
        echo "${BACKUP_PREFIX}_${backup_type}_full_${hostname}_${timestamp}"
    fi
}

# Create database backup
backup_database() {
    log_info "Creating database backup..."
    
    local backup_file="$1"
    local temp_dir=$(mktemp -d)
    
    # PostgreSQL backup
    if command_exists "pg_dump" && [[ -n "${POSTGRES_PASSWORD:-}" ]]; then
        log_info "Backing up PostgreSQL database..."
        
        export PGPASSWORD="$POSTGRES_PASSWORD"
        local pg_host="${POSTGRES_HOST:-localhost}"
        local pg_port="${POSTGRES_PORT:-5432}"
        local pg_user="${POSTGRES_USER:-aegis}"
        local pg_db="${POSTGRES_DB:-aegis}"
        
        if [[ "$DRY_RUN" != true ]]; then
            pg_dump -h "$pg_host" -p "$pg_port" -U "$pg_user" -d "$pg_db" \
                --verbose --clean --if-exists --create \
                > "$temp_dir/postgres_backup.sql" 2>>"$LOG_FILE"
            
            if [[ $? -eq 0 ]]; then
                log_success "PostgreSQL backup completed"
            else
                log_error "PostgreSQL backup failed"
                return 1
            fi
        else
            log_info "[DRY RUN] Would backup PostgreSQL database"
        fi
        
        unset PGPASSWORD
    else
        log_warning "PostgreSQL backup skipped (pg_dump not available or password not set)"
    fi
    
    # Redis backup
    if command_exists "redis-cli" && [[ -n "${REDIS_PASSWORD:-}" ]]; then
        log_info "Backing up Redis database..."
        
        local redis_host="${REDIS_HOST:-localhost}"
        local redis_port="${REDIS_PORT:-6379}"
        
        if [[ "$DRY_RUN" != true ]]; then
            redis-cli -h "$redis_host" -p "$redis_port" -a "$REDIS_PASSWORD" \
                --rdb "$temp_dir/redis_backup.rdb" 2>>"$LOG_FILE"
            
            if [[ $? -eq 0 ]]; then
                log_success "Redis backup completed"
            else
                log_error "Redis backup failed"
                return 1
            fi
        else
            log_info "[DRY RUN] Would backup Redis database"
        fi
    else
        log_warning "Redis backup skipped (redis-cli not available or password not set)"
    fi
    
    # Create archive
    if [[ "$DRY_RUN" != true ]]; then
        tar -czf "$backup_file" -C "$temp_dir" . 2>>"$LOG_FILE"
        rm -rf "$temp_dir"
    fi
    
    log_success "Database backup created: $backup_file"
}

# Create configuration backup
backup_config() {
    log_info "Creating configuration backup..."
    
    local backup_file="$1"
    local temp_dir=$(mktemp -d)
    
    # Configuration files
    local config_files=(
        "$AEGIS_DIR/config"
        "$AEGIS_DIR/.env"
        "$AEGIS_DIR/docker-compose.yml"
        "$AEGIS_DIR/docker-compose.dev.yml"
        "$AEGIS_DIR/Dockerfile"
        "$AEGIS_DIR/requirements.txt"
        "$AEGIS_DIR/package.json"
    )
    
    # Certificates and keys
    local cert_dirs=(
        "$AEGIS_DIR/certs"
        "$AEGIS_DIR/keys"
    )
    
    if [[ "$DRY_RUN" != true ]]; then
        # Copy configuration files
        for file in "${config_files[@]}"; do
            if [[ -e "$file" ]]; then
                cp -r "$file" "$temp_dir/" 2>>"$LOG_FILE"
                log_info "Backed up: $(basename "$file")"
            fi
        done
        
        # Copy certificate directories
        for dir in "${cert_dirs[@]}"; do
            if [[ -d "$dir" ]]; then
                cp -r "$dir" "$temp_dir/" 2>>"$LOG_FILE"
                log_info "Backed up: $(basename "$dir")"
            fi
        done
        
        # Create archive
        tar -czf "$backup_file" -C "$temp_dir" . 2>>"$LOG_FILE"
        rm -rf "$temp_dir"
    else
        log_info "[DRY RUN] Would backup configuration files and certificates"
    fi
    
    log_success "Configuration backup created: $backup_file"
}

# Create logs backup
backup_logs() {
    log_info "Creating logs backup..."
    
    local backup_file="$1"
    local temp_dir=$(mktemp -d)
    
    # Log directories
    local log_dirs=(
        "$AEGIS_DIR/logs"
        "/var/log/aegis"
    )
    
    if [[ "$DRY_RUN" != true ]]; then
        for dir in "${log_dirs[@]}"; do
            if [[ -d "$dir" ]]; then
                cp -r "$dir" "$temp_dir/" 2>>"$LOG_FILE"
                log_info "Backed up: $(basename "$dir")"
            fi
        done
        
        # Create archive
        tar -czf "$backup_file" -C "$temp_dir" . 2>>"$LOG_FILE"
        rm -rf "$temp_dir"
    else
        log_info "[DRY RUN] Would backup log files"
    fi
    
    log_success "Logs backup created: $backup_file"
}

# Create Docker volumes backup
backup_docker() {
    log_info "Creating Docker volumes backup..."
    
    local backup_file="$1"
    local temp_dir=$(mktemp -d)
    
    if ! command_exists "docker"; then
        log_warning "Docker not available, skipping Docker backup"
        return
    fi
    
    # Get Docker volumes
    local volumes=$(docker volume ls -q | grep aegis 2>/dev/null || true)
    
    if [[ -n "$volumes" ]]; then
        if [[ "$DRY_RUN" != true ]]; then
            for volume in $volumes; do
                log_info "Backing up Docker volume: $volume"
                
                # Create temporary container to access volume
                docker run --rm -v "$volume:/data" -v "$temp_dir:/backup" \
                    alpine tar -czf "/backup/${volume}.tar.gz" -C /data . 2>>"$LOG_FILE"
                
                if [[ $? -eq 0 ]]; then
                    log_success "Volume $volume backed up"
                else
                    log_error "Failed to backup volume $volume"
                fi
            done
            
            # Create final archive
            tar -czf "$backup_file" -C "$temp_dir" . 2>>"$LOG_FILE"
            rm -rf "$temp_dir"
        else
            log_info "[DRY RUN] Would backup Docker volumes: $volumes"
        fi
    else
        log_warning "No AEGIS Docker volumes found"
    fi
    
    log_success "Docker backup created: $backup_file"
}

# Create full backup
backup_full() {
    log_info "Creating full system backup..."
    
    local backup_file="$1"
    local temp_dir=$(mktemp -d)
    
    # Create individual backups
    backup_database "$temp_dir/database.tar.gz"
    backup_config "$temp_dir/config.tar.gz"
    backup_logs "$temp_dir/logs.tar.gz"
    backup_docker "$temp_dir/docker.tar.gz"
    
    # Create metadata file
    cat > "$temp_dir/backup_metadata.json" << EOF
{
    "backup_type": "full",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hostname": "$(hostname)",
    "aegis_version": "$(cat "$AEGIS_DIR/VERSION" 2>/dev/null || echo "unknown")",
    "components": [
        "database",
        "config",
        "logs",
        "docker"
    ],
    "compression_level": $COMPRESSION_LEVEL,
    "encrypted": $ENCRYPTION_ENABLED
}
EOF
    
    if [[ "$DRY_RUN" != true ]]; then
        # Create final archive
        tar -czf "$backup_file" -C "$temp_dir" . 2>>"$LOG_FILE"
        rm -rf "$temp_dir"
    fi
    
    log_success "Full backup created: $backup_file"
}

# Encrypt backup file
encrypt_backup() {
    local backup_file="$1"
    local encrypted_file="${backup_file}.enc"
    
    if [[ "$ENCRYPTION_ENABLED" != true ]]; then
        return
    fi
    
    log_info "Encrypting backup file..."
    
    local encryption_key="${BACKUP_ENCRYPTION_KEY:-}"
    if [[ -z "$encryption_key" ]]; then
        # Generate random key if not provided
        encryption_key=$(openssl rand -hex 32)
        log_warning "Generated random encryption key: $encryption_key"
        log_warning "Save this key to decrypt the backup later!"
    fi
    
    if [[ "$DRY_RUN" != true ]]; then
        openssl enc -aes-256-cbc -salt -in "$backup_file" -out "$encrypted_file" -k "$encryption_key" 2>>"$LOG_FILE"
        
        if [[ $? -eq 0 ]]; then
            rm "$backup_file"
            log_success "Backup encrypted: $encrypted_file"
            echo "$encrypted_file"
        else
            log_error "Encryption failed"
            echo "$backup_file"
        fi
    else
        log_info "[DRY RUN] Would encrypt backup file"
        echo "$backup_file"
    fi
}

# Upload to remote location
upload_remote() {
    local backup_file="$1"
    
    if [[ "$REMOTE_BACKUP" != true ]]; then
        return
    fi
    
    log_info "Uploading backup to remote location..."
    
    local remote_url="${REMOTE_BACKUP_URL:-}"
    if [[ -z "$remote_url" ]]; then
        log_warning "Remote backup URL not configured"
        return
    fi
    
    if [[ "$DRY_RUN" != true ]]; then
        case "$remote_url" in
            rsync://*)
                rsync -avz --progress "$backup_file" "$remote_url" 2>>"$LOG_FILE"
                ;;
            s3://*)
                if command_exists "aws"; then
                    aws s3 cp "$backup_file" "$remote_url" 2>>"$LOG_FILE"
                else
                    log_error "AWS CLI not available for S3 upload"
                fi
                ;;
            *)
                log_error "Unsupported remote backup URL: $remote_url"
                ;;
        esac
        
        if [[ $? -eq 0 ]]; then
            log_success "Backup uploaded to remote location"
        else
            log_error "Remote upload failed"
        fi
    else
        log_info "[DRY RUN] Would upload backup to: $remote_url"
    fi
}

# Create backup
create_backup() {
    log_header "Starting backup process..."
    
    local filename=$(generate_backup_filename "$BACKUP_TYPE")
    local backup_file="$BACKUP_DIR/${filename}.tar.gz"
    
    log_info "Backup type: $BACKUP_TYPE"
    log_info "Backup file: $backup_file"
    
    # Create backup based on type
    case "$BACKUP_TYPE" in
        full)
            backup_full "$backup_file"
            ;;
        database)
            backup_database "$backup_file"
            ;;
        config)
            backup_config "$backup_file"
            ;;
        logs)
            backup_logs "$backup_file"
            ;;
        docker)
            backup_docker "$backup_file"
            ;;
        *)
            log_error "Unknown backup type: $BACKUP_TYPE"
            exit 1
            ;;
    esac
    
    # Encrypt if enabled
    backup_file=$(encrypt_backup "$backup_file")
    
    # Upload to remote if enabled
    upload_remote "$backup_file"
    
    # Calculate backup size
    if [[ -f "$backup_file" ]]; then
        local backup_size=$(du -h "$backup_file" | cut -f1)
        log_success "Backup completed successfully"
        log_info "Backup size: $backup_size"
        log_info "Backup location: $backup_file"
    fi
}

# List available backups
list_backups() {
    log_header "Available backups:"
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log_warning "Backup directory does not exist: $BACKUP_DIR"
        return
    fi
    
    local backups=$(find "$BACKUP_DIR" -name "${BACKUP_PREFIX}_*" -type f | sort -r)
    
    if [[ -z "$backups" ]]; then
        log_info "No backups found"
        return
    fi
    
    printf "%-50s %-15s %-10s %-20s\n" "BACKUP FILE" "TYPE" "SIZE" "DATE"
    printf "%-50s %-15s %-10s %-20s\n" "$(printf '%*s' 50 | tr ' ' '-')" "$(printf '%*s' 15 | tr ' ' '-')" "$(printf '%*s' 10 | tr ' ' '-')" "$(printf '%*s' 20 | tr ' ' '-')"
    
    while IFS= read -r backup_file; do
        local filename=$(basename "$backup_file")
        local backup_type=$(echo "$filename" | cut -d'_' -f3)
        local backup_size=$(du -h "$backup_file" | cut -f1)
        local backup_date=$(stat -c %y "$backup_file" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
        
        printf "%-50s %-15s %-10s %-20s\n" "$filename" "$backup_type" "$backup_size" "$backup_date"
    done <<< "$backups"
}

# Verify backup integrity
verify_backup() {
    local backup_file="${BACKUP_FILE:-}"
    
    if [[ -z "$backup_file" ]]; then
        log_error "No backup file specified for verification"
        exit 1
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        backup_file="$BACKUP_DIR/$backup_file"
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log_header "Verifying backup: $backup_file"
    
    # Check if encrypted
    if [[ "$backup_file" == *.enc ]]; then
        log_info "Backup is encrypted, attempting to decrypt for verification..."
        
        local encryption_key="${BACKUP_ENCRYPTION_KEY:-}"
        if [[ -z "$encryption_key" ]]; then
            log_error "Encryption key required for verification"
            exit 1
        fi
        
        local temp_file=$(mktemp)
        openssl enc -aes-256-cbc -d -in "$backup_file" -out "$temp_file" -k "$encryption_key" 2>>"$LOG_FILE"
        
        if [[ $? -ne 0 ]]; then
            log_error "Failed to decrypt backup for verification"
            rm -f "$temp_file"
            exit 1
        fi
        
        backup_file="$temp_file"
    fi
    
    # Verify archive integrity
    log_info "Checking archive integrity..."
    tar -tzf "$backup_file" >/dev/null 2>>"$LOG_FILE"
    
    if [[ $? -eq 0 ]]; then
        log_success "Backup archive is valid"
        
        # List contents if verbose
        if [[ "$VERBOSE" == true ]]; then
            log_info "Backup contents:"
            tar -tzf "$backup_file" | head -20
        fi
    else
        log_error "Backup archive is corrupted"
        exit 1
    fi
    
    # Clean up temporary file
    if [[ "$backup_file" == /tmp/* ]]; then
        rm -f "$backup_file"
    fi
}

# Restore from backup
restore_backup() {
    local backup_file="${BACKUP_FILE:-}"
    
    if [[ -z "$backup_file" ]]; then
        log_error "No backup file specified for restore"
        exit 1
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        backup_file="$BACKUP_DIR/$backup_file"
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log_header "Restoring from backup: $backup_file"
    log_warning "This will overwrite existing data. Continue? (y/N)"
    
    if [[ "$DRY_RUN" != true ]]; then
        read -r confirmation
        if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
            log_info "Restore cancelled"
            exit 0
        fi
    fi
    
    # Decrypt if necessary
    local restore_file="$backup_file"
    if [[ "$backup_file" == *.enc ]]; then
        log_info "Decrypting backup..."
        
        local encryption_key="${BACKUP_ENCRYPTION_KEY:-}"
        if [[ -z "$encryption_key" ]]; then
            log_error "Encryption key required for restore"
            exit 1
        fi
        
        restore_file=$(mktemp)
        openssl enc -aes-256-cbc -d -in "$backup_file" -out "$restore_file" -k "$encryption_key" 2>>"$LOG_FILE"
        
        if [[ $? -ne 0 ]]; then
            log_error "Failed to decrypt backup"
            rm -f "$restore_file"
            exit 1
        fi
    fi
    
    # Extract and restore
    local temp_dir=$(mktemp -d)
    
    if [[ "$DRY_RUN" != true ]]; then
        tar -xzf "$restore_file" -C "$temp_dir" 2>>"$LOG_FILE"
        
        # Restore based on backup contents
        if [[ -f "$temp_dir/backup_metadata.json" ]]; then
            log_info "Found backup metadata, performing structured restore..."
            
            # Restore configuration
            if [[ -f "$temp_dir/config.tar.gz" ]]; then
                log_info "Restoring configuration..."
                tar -xzf "$temp_dir/config.tar.gz" -C "$AEGIS_DIR" 2>>"$LOG_FILE"
            fi
            
            # Restore databases (requires manual intervention)
            if [[ -f "$temp_dir/database.tar.gz" ]]; then
                log_warning "Database restore requires manual intervention"
                log_info "Database backup extracted to: $temp_dir/database"
                tar -xzf "$temp_dir/database.tar.gz" -C "$temp_dir/database" 2>>"$LOG_FILE"
            fi
        else
            log_info "Performing direct restore..."
            cp -r "$temp_dir"/* "$AEGIS_DIR/" 2>>"$LOG_FILE"
        fi
        
        log_success "Restore completed"
    else
        log_info "[DRY RUN] Would restore from backup"
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    if [[ "$restore_file" != "$backup_file" ]]; then
        rm -f "$restore_file"
    fi
}

# Cleanup old backups
cleanup_backups() {
    log_header "Cleaning up old backups..."
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log_warning "Backup directory does not exist: $BACKUP_DIR"
        return
    fi
    
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%s)
    local deleted_count=0
    local total_size=0
    
    while IFS= read -r backup_file; do
        local file_date=$(stat -c %Y "$backup_file" 2>/dev/null || echo "0")
        
        if [[ $file_date -lt $cutoff_date ]]; then
            local file_size=$(stat -c %s "$backup_file" 2>/dev/null || echo "0")
            total_size=$((total_size + file_size))
            
            if [[ "$DRY_RUN" != true ]]; then
                rm "$backup_file"
                log_info "Deleted old backup: $(basename "$backup_file")"
            else
                log_info "[DRY RUN] Would delete: $(basename "$backup_file")"
            fi
            
            deleted_count=$((deleted_count + 1))
        fi
    done < <(find "$BACKUP_DIR" -name "${BACKUP_PREFIX}_*" -type f)
    
    if [[ $deleted_count -gt 0 ]]; then
        local size_mb=$((total_size / 1024 / 1024))
        log_success "Cleaned up $deleted_count old backups (${size_mb}MB freed)"
    else
        log_info "No old backups to clean up"
    fi
}

# Main function
main() {
    # Parse arguments
    parse_arguments "$@"
    
    # Initialize
    initialize_backup
    
    # Execute command
    if [[ "$LIST_BACKUPS" == true ]]; then
        list_backups
    elif [[ "$RESTORE_MODE" == true ]]; then
        restore_backup
    elif [[ "$CLEANUP_MODE" == true ]]; then
        cleanup_backups
    elif [[ "$VERIFY_MODE" == true ]]; then
        verify_backup
    else
        create_backup
    fi
}

# Run main function with all arguments
main "$@"