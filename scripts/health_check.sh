#!/bin/bash

# AEGIS Framework - Health Check Script
# Comprehensive system health verification

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/health_check.log"
TIMEOUT=30
VERBOSE=false
JSON_OUTPUT=false
ALERT_THRESHOLD=80

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Health check results
HEALTH_STATUS="healthy"
ISSUES=()
WARNINGS=()

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

# Check Docker services
check_docker_services() {
    log "INFO" "Checking Docker services..."
    
    if ! command_exists docker; then
        ISSUES+=("Docker not installed")
        HEALTH_STATUS="unhealthy"
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        ISSUES+=("Docker daemon not running")
        HEALTH_STATUS="unhealthy"
        return 1
    fi
    
    # Check if docker-compose is available
    if command_exists docker-compose; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        ISSUES+=("Docker Compose not available")
        HEALTH_STATUS="unhealthy"
        return 1
    fi
    
    # Check running containers
    local containers
    containers=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(aegis|redis|postgres|prometheus|grafana|nginx)" || true)
    
    if [[ -z "$containers" ]]; then
        WARNINGS+=("No AEGIS containers running")
        return 0
    fi
    
    # Check container health
    while IFS=$'\t' read -r name status; do
        if [[ "$name" == "NAMES" ]]; then continue; fi
        
        if [[ "$status" =~ "Up" ]]; then
            log "SUCCESS" "Container $name is running"
        else
            ISSUES+=("Container $name is not healthy: $status")
            HEALTH_STATUS="degraded"
        fi
    done <<< "$containers"
    
    log "SUCCESS" "Docker services check completed"
}

# Check system resources
check_system_resources() {
    log "INFO" "Checking system resources..."
    
    # CPU usage
    if command_exists top; then
        local cpu_usage
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
        cpu_usage=${cpu_usage%.*}
        
        if [[ "$cpu_usage" -gt "$ALERT_THRESHOLD" ]]; then
            WARNINGS+=("High CPU usage: ${cpu_usage}%")
        fi
        log "INFO" "CPU usage: ${cpu_usage}%"
    fi
    
    # Memory usage
    if command_exists free; then
        local mem_info
        mem_info=$(free | grep Mem)
        local total=$(echo "$mem_info" | awk '{print $2}')
        local used=$(echo "$mem_info" | awk '{print $3}')
        local mem_percent=$((used * 100 / total))
        
        if [[ "$mem_percent" -gt "$ALERT_THRESHOLD" ]]; then
            WARNINGS+=("High memory usage: ${mem_percent}%")
        fi
        log "INFO" "Memory usage: ${mem_percent}%"
    fi
    
    # Disk usage
    local disk_usage
    disk_usage=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [[ "$disk_usage" -gt "$ALERT_THRESHOLD" ]]; then
        WARNINGS+=("High disk usage: ${disk_usage}%")
    fi
    log "INFO" "Disk usage: ${disk_usage}%"
    
    log "SUCCESS" "System resources check completed"
}

# Check network connectivity
check_network() {
    log "INFO" "Checking network connectivity..."
    
    # Check localhost ports
    local ports=(5000 8080 8181 8051 8052 3737 5432 6379 9090 3000)
    
    for port in "${ports[@]}"; do
        if command_exists nc; then
            if nc -z localhost "$port" 2>/dev/null; then
                log "SUCCESS" "Port $port is accessible"
            else
                log "INFO" "Port $port is not accessible (service may be down)"
            fi
        elif command_exists telnet; then
            if timeout 1 telnet localhost "$port" 2>/dev/null | grep -q "Connected"; then
                log "SUCCESS" "Port $port is accessible"
            else
                log "INFO" "Port $port is not accessible (service may be down)"
            fi
        fi
    done
    
    # Check external connectivity
    if command_exists curl; then
        if curl -s --max-time 5 https://httpbin.org/ip >/dev/null; then
            log "SUCCESS" "External connectivity working"
        else
            WARNINGS+=("External connectivity issues")
        fi
    fi
    
    log "SUCCESS" "Network connectivity check completed"
}

# Check AEGIS services
check_aegis_services() {
    log "INFO" "Checking AEGIS services..."
    
    # Check main application
    if curl -s --max-time 5 http://localhost:5000/health >/dev/null 2>&1; then
        log "SUCCESS" "AEGIS main service is responding"
    else
        ISSUES+=("AEGIS main service not responding")
        HEALTH_STATUS="degraded"
    fi
    
    # Check API server
    if curl -s --max-time 5 http://localhost:8080/health >/dev/null 2>&1; then
        log "SUCCESS" "AEGIS API server is responding"
    else
        WARNINGS+=("AEGIS API server not responding")
    fi
    
    # Check if TOR is running
    if pgrep -f "tor" >/dev/null; then
        log "SUCCESS" "TOR service is running"
    else
        log "INFO" "TOR service is not running (optional)"
    fi
    
    log "SUCCESS" "AEGIS services check completed"
}

# Check log files
check_logs() {
    log "INFO" "Checking log files..."
    
    local log_dir="${PROJECT_ROOT}/logs"
    
    if [[ ! -d "$log_dir" ]]; then
        WARNINGS+=("Log directory does not exist")
        return 0
    fi
    
    # Check for recent errors
    local error_count=0
    if [[ -f "${log_dir}/aegis.log" ]]; then
        error_count=$(grep -c "ERROR" "${log_dir}/aegis.log" 2>/dev/null || echo "0")
        if [[ "$error_count" -gt 10 ]]; then
            WARNINGS+=("High error count in logs: $error_count")
        fi
    fi
    
    # Check log file sizes
    find "$log_dir" -name "*.log" -size +100M 2>/dev/null | while read -r large_log; do
        WARNINGS+=("Large log file: $(basename "$large_log")")
    done
    
    log "SUCCESS" "Log files check completed"
}

# Check security
check_security() {
    log "INFO" "Checking security..."
    
    # Check for default passwords
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        if grep -q "password123\|admin123\|secret123" "${PROJECT_ROOT}/.env" 2>/dev/null; then
            ISSUES+=("Default passwords detected in .env file")
            HEALTH_STATUS="unhealthy"
        fi
    fi
    
    # Check file permissions
    local sensitive_files=(".env" "config/ssl" "config/keys")
    for file in "${sensitive_files[@]}"; do
        local full_path="${PROJECT_ROOT}/${file}"
        if [[ -e "$full_path" ]]; then
            local perms
            perms=$(stat -c "%a" "$full_path" 2>/dev/null || stat -f "%A" "$full_path" 2>/dev/null || echo "unknown")
            if [[ "$perms" != "600" && "$perms" != "700" ]]; then
                WARNINGS+=("Insecure permissions on $file: $perms")
            fi
        fi
    done
    
    # Check for SSL certificates
    if [[ -d "${PROJECT_ROOT}/config/ssl" ]]; then
        local cert_files
        cert_files=$(find "${PROJECT_ROOT}/config/ssl" -name "*.crt" -o -name "*.pem" 2>/dev/null | wc -l)
        if [[ "$cert_files" -eq 0 ]]; then
            WARNINGS+=("No SSL certificates found")
        fi
    fi
    
    log "SUCCESS" "Security check completed"
}

# Generate report
generate_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [[ "$JSON_OUTPUT" == "true" ]]; then
        # JSON output
        cat << EOF
{
    "timestamp": "$timestamp",
    "status": "$HEALTH_STATUS",
    "issues": [$(printf '"%s",' "${ISSUES[@]}" | sed 's/,$//')],
    "warnings": [$(printf '"%s",' "${WARNINGS[@]}" | sed 's/,$//')],
    "summary": {
        "total_issues": ${#ISSUES[@]},
        "total_warnings": ${#WARNINGS[@]}
    }
}
EOF
    else
        # Human readable output
        echo
        echo "========================================="
        echo "AEGIS Framework Health Check Report"
        echo "========================================="
        echo "Timestamp: $timestamp"
        echo "Overall Status: $HEALTH_STATUS"
        echo
        
        if [[ ${#ISSUES[@]} -gt 0 ]]; then
            echo -e "${RED}Issues Found:${NC}"
            for issue in "${ISSUES[@]}"; do
                echo "  ❌ $issue"
            done
            echo
        fi
        
        if [[ ${#WARNINGS[@]} -gt 0 ]]; then
            echo -e "${YELLOW}Warnings:${NC}"
            for warning in "${WARNINGS[@]}"; do
                echo "  ⚠️  $warning"
            done
            echo
        fi
        
        if [[ ${#ISSUES[@]} -eq 0 && ${#WARNINGS[@]} -eq 0 ]]; then
            echo -e "${GREEN}✅ All systems are healthy!${NC}"
        fi
        
        echo "Summary:"
        echo "  - Issues: ${#ISSUES[@]}"
        echo "  - Warnings: ${#WARNINGS[@]}"
        echo "========================================="
    fi
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

AEGIS Framework Health Check Script

OPTIONS:
    -v, --verbose       Enable verbose output
    -j, --json          Output results in JSON format
    -t, --timeout SEC   Set timeout for checks (default: 30)
    -h, --help          Show this help message

EXAMPLES:
    $0                  Run basic health check
    $0 -v               Run with verbose output
    $0 -j               Output results in JSON format
    $0 -v -t 60         Run with verbose output and 60s timeout

EXIT CODES:
    0   All checks passed (healthy)
    1   Critical issues found (unhealthy)
    2   Warnings found (degraded)
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
            -j|--json)
                JSON_OUTPUT=true
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
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
    
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Start health check
    log "INFO" "Starting AEGIS Framework health check..."
    
    # Run all checks
    check_docker_services
    check_system_resources
    check_network
    check_aegis_services
    check_logs
    check_security
    
    # Generate report
    generate_report
    
    # Exit with appropriate code
    case "$HEALTH_STATUS" in
        "healthy")
            log "SUCCESS" "Health check completed - All systems healthy"
            exit 0
            ;;
        "degraded")
            log "WARN" "Health check completed - System degraded"
            exit 2
            ;;
        "unhealthy")
            log "ERROR" "Health check completed - Critical issues found"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"