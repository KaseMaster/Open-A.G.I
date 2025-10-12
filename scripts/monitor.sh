#!/bin/bash

# AEGIS Framework - Monitoring Script
# Advanced Encrypted Governance and Intelligence System
# This script provides comprehensive monitoring and alerting

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
MONITOR_INTERVAL=30
LOG_FILE="$AEGIS_DIR/logs/monitor.log"
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90
ALERT_THRESHOLD_RESPONSE_TIME=5000
MAX_LOG_SIZE=100M
RETENTION_DAYS=30

# Default values
DAEMON_MODE=false
VERBOSE=false
CHECK_SERVICES=true
CHECK_RESOURCES=true
CHECK_NETWORK=true
CHECK_SECURITY=true
SEND_ALERTS=true

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
    echo -e "${PURPLE}[AEGIS MONITOR]${NC} $message"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Show help
show_help() {
    cat << EOF
AEGIS Framework Monitoring Script

Usage: $0 [OPTIONS]

OPTIONS:
    -d, --daemon            Run in daemon mode (continuous monitoring)
    -v, --verbose           Enable verbose output
    -i, --interval SECONDS  Set monitoring interval (default: 30)
    --no-services          Skip service health checks
    --no-resources         Skip resource monitoring
    --no-network           Skip network monitoring
    --no-security          Skip security monitoring
    --no-alerts            Disable alert notifications
    -h, --help             Show this help message

MONITORING CATEGORIES:
    Services    - Docker containers, processes, health endpoints
    Resources   - CPU, memory, disk usage
    Network     - Connectivity, latency, port availability
    Security    - Failed logins, suspicious activity, certificate expiry

EXAMPLES:
    $0                      # Run single monitoring check
    $0 --daemon             # Run continuous monitoring
    $0 --daemon --interval 60  # Monitor every 60 seconds
    $0 --no-alerts          # Monitor without sending alerts

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--daemon)
                DAEMON_MODE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -i|--interval)
                MONITOR_INTERVAL="$2"
                shift 2
                ;;
            --no-services)
                CHECK_SERVICES=false
                shift
                ;;
            --no-resources)
                CHECK_RESOURCES=false
                shift
                ;;
            --no-network)
                CHECK_NETWORK=false
                shift
                ;;
            --no-security)
                CHECK_SECURITY=false
                shift
                ;;
            --no-alerts)
                SEND_ALERTS=false
                shift
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

# Initialize monitoring
initialize_monitoring() {
    log_header "Initializing AEGIS monitoring..."
    
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Load environment variables
    if [[ -f "$AEGIS_DIR/.env" ]]; then
        set -a
        source "$AEGIS_DIR/.env"
        set +a
    fi
    
    # Check required tools
    local required_tools=("docker" "curl" "ps" "df" "free")
    for tool in "${required_tools[@]}"; do
        if ! command_exists "$tool"; then
            log_warning "$tool not found, some checks may be skipped"
        fi
    done
    
    log_success "Monitoring initialized"
}

# Check Docker services
check_docker_services() {
    if [[ "$CHECK_SERVICES" != true ]]; then
        return
    fi
    
    log_info "Checking Docker services..."
    
    if ! command_exists docker; then
        log_warning "Docker not available, skipping container checks"
        return
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        send_alert "Docker daemon is not running"
        return
    fi
    
    # Check AEGIS containers
    local containers=(
        "aegis-core"
        "aegis-redis"
        "aegis-postgres"
        "aegis-prometheus"
        "aegis-grafana"
        "aegis-nginx"
    )
    
    for container in "${containers[@]}"; do
        local status=$(docker ps --filter "name=$container" --format "{{.Status}}" 2>/dev/null || echo "Not found")
        
        if [[ "$status" == "Not found" ]]; then
            log_warning "Container $container is not running"
            if [[ "$SEND_ALERTS" == true ]]; then
                send_alert "Container $container is not running"
            fi
        elif [[ "$status" =~ ^Up ]]; then
            log_success "Container $container is healthy: $status"
        else
            log_error "Container $container has issues: $status"
            if [[ "$SEND_ALERTS" == true ]]; then
                send_alert "Container $container has issues: $status"
            fi
        fi
    done
}

# Check health endpoints
check_health_endpoints() {
    if [[ "$CHECK_SERVICES" != true ]]; then
        return
    fi
    
    log_info "Checking health endpoints..."
    
    local endpoints=(
        "http://localhost:8080/health:AEGIS Core"
        "http://localhost:9090/api/v1/query?query=up:Prometheus"
        "http://localhost:3000/api/health:Grafana"
        "http://localhost:6379/ping:Redis"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        local endpoint="${endpoint_info%:*}"
        local service="${endpoint_info#*:}"
        
        local start_time=$(date +%s%3N)
        local response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 --max-time 10 "$endpoint" 2>/dev/null || echo "000")
        local end_time=$(date +%s%3N)
        local response_time=$((end_time - start_time))
        
        if [[ "$response_code" == "200" ]]; then
            log_success "$service health check passed (${response_time}ms)"
            
            if [[ $response_time -gt $ALERT_THRESHOLD_RESPONSE_TIME ]]; then
                log_warning "$service response time is high: ${response_time}ms"
                if [[ "$SEND_ALERTS" == true ]]; then
                    send_alert "$service response time is high: ${response_time}ms"
                fi
            fi
        else
            log_error "$service health check failed (HTTP $response_code)"
            if [[ "$SEND_ALERTS" == true ]]; then
                send_alert "$service health check failed (HTTP $response_code)"
            fi
        fi
    done
}

# Check system resources
check_system_resources() {
    if [[ "$CHECK_RESOURCES" != true ]]; then
        return
    fi
    
    log_info "Checking system resources..."
    
    # Check CPU usage
    if command_exists top; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' | cut -d'%' -f1)
        if [[ -n "$cpu_usage" ]]; then
            cpu_usage=${cpu_usage%.*}  # Remove decimal part
            log_info "CPU usage: ${cpu_usage}%"
            
            if [[ $cpu_usage -gt $ALERT_THRESHOLD_CPU ]]; then
                log_warning "High CPU usage: ${cpu_usage}%"
                if [[ "$SEND_ALERTS" == true ]]; then
                    send_alert "High CPU usage: ${cpu_usage}%"
                fi
            fi
        fi
    fi
    
    # Check memory usage
    if command_exists free; then
        local memory_info=$(free | grep Mem)
        local total_mem=$(echo "$memory_info" | awk '{print $2}')
        local used_mem=$(echo "$memory_info" | awk '{print $3}')
        local memory_usage=$((used_mem * 100 / total_mem))
        
        log_info "Memory usage: ${memory_usage}% ($(($used_mem / 1024))MB / $(($total_mem / 1024))MB)"
        
        if [[ $memory_usage -gt $ALERT_THRESHOLD_MEMORY ]]; then
            log_warning "High memory usage: ${memory_usage}%"
            if [[ "$SEND_ALERTS" == true ]]; then
                send_alert "High memory usage: ${memory_usage}%"
            fi
        fi
    fi
    
    # Check disk usage
    if command_exists df; then
        while IFS= read -r line; do
            local filesystem=$(echo "$line" | awk '{print $1}')
            local usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')
            local mount=$(echo "$line" | awk '{print $6}')
            
            if [[ "$usage" =~ ^[0-9]+$ ]]; then
                log_info "Disk usage $mount: ${usage}%"
                
                if [[ $usage -gt $ALERT_THRESHOLD_DISK ]]; then
                    log_warning "High disk usage on $mount: ${usage}%"
                    if [[ "$SEND_ALERTS" == true ]]; then
                        send_alert "High disk usage on $mount: ${usage}%"
                    fi
                fi
            fi
        done < <(df -h | grep -E '^/dev/')
    fi
}

# Check network connectivity
check_network_connectivity() {
    if [[ "$CHECK_NETWORK" != true ]]; then
        return
    fi
    
    log_info "Checking network connectivity..."
    
    # Check internet connectivity
    if ping -c 1 -W 5 8.8.8.8 >/dev/null 2>&1; then
        log_success "Internet connectivity: OK"
    else
        log_error "Internet connectivity: FAILED"
        if [[ "$SEND_ALERTS" == true ]]; then
            send_alert "Internet connectivity failed"
        fi
    fi
    
    # Check DNS resolution
    if nslookup google.com >/dev/null 2>&1; then
        log_success "DNS resolution: OK"
    else
        log_error "DNS resolution: FAILED"
        if [[ "$SEND_ALERTS" == true ]]; then
            send_alert "DNS resolution failed"
        fi
    fi
    
    # Check critical ports
    local ports=(
        "8080:AEGIS Core"
        "5432:PostgreSQL"
        "6379:Redis"
        "9090:Prometheus"
        "3000:Grafana"
    )
    
    for port_info in "${ports[@]}"; do
        local port="${port_info%:*}"
        local service="${port_info#*:}"
        
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            log_success "Port $port ($service): LISTENING"
        else
            log_warning "Port $port ($service): NOT LISTENING"
            if [[ "$SEND_ALERTS" == true ]]; then
                send_alert "Port $port ($service) is not listening"
            fi
        fi
    done
}

# Check security metrics
check_security_metrics() {
    if [[ "$CHECK_SECURITY" != true ]]; then
        return
    fi
    
    log_info "Checking security metrics..."
    
    # Check failed login attempts
    if [[ -f "/var/log/auth.log" ]]; then
        local failed_logins=$(grep "Failed password" /var/log/auth.log | grep "$(date '+%b %d')" | wc -l)
        log_info "Failed login attempts today: $failed_logins"
        
        if [[ $failed_logins -gt 10 ]]; then
            log_warning "High number of failed login attempts: $failed_logins"
            if [[ "$SEND_ALERTS" == true ]]; then
                send_alert "High number of failed login attempts: $failed_logins"
            fi
        fi
    fi
    
    # Check SSL certificate expiry
    if [[ -f "$AEGIS_DIR/certs/server.crt" ]]; then
        local cert_expiry=$(openssl x509 -in "$AEGIS_DIR/certs/server.crt" -noout -enddate 2>/dev/null | cut -d= -f2)
        if [[ -n "$cert_expiry" ]]; then
            local expiry_timestamp=$(date -d "$cert_expiry" +%s 2>/dev/null || echo "0")
            local current_timestamp=$(date +%s)
            local days_until_expiry=$(( (expiry_timestamp - current_timestamp) / 86400 ))
            
            log_info "SSL certificate expires in $days_until_expiry days"
            
            if [[ $days_until_expiry -lt 30 ]]; then
                log_warning "SSL certificate expires soon: $days_until_expiry days"
                if [[ "$SEND_ALERTS" == true ]]; then
                    send_alert "SSL certificate expires in $days_until_expiry days"
                fi
            fi
        fi
    fi
    
    # Check for suspicious processes
    local suspicious_processes=$(ps aux | grep -E "(nc|netcat|nmap|tcpdump)" | grep -v grep | wc -l)
    if [[ $suspicious_processes -gt 0 ]]; then
        log_warning "Suspicious network processes detected: $suspicious_processes"
        if [[ "$SEND_ALERTS" == true ]]; then
            send_alert "Suspicious network processes detected"
        fi
    fi
}

# Check log files
check_log_files() {
    log_info "Checking log files..."
    
    # Check log file sizes
    local log_dirs=("$AEGIS_DIR/logs" "/var/log")
    
    for log_dir in "${log_dirs[@]}"; do
        if [[ -d "$log_dir" ]]; then
            while IFS= read -r logfile; do
                local size=$(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile" 2>/dev/null || echo "0")
                local size_mb=$((size / 1024 / 1024))
                
                if [[ $size_mb -gt 100 ]]; then
                    log_warning "Large log file: $logfile (${size_mb}MB)"
                fi
            done < <(find "$log_dir" -name "*.log" -type f 2>/dev/null)
        fi
    done
    
    # Check for error patterns in AEGIS logs
    if [[ -f "$AEGIS_DIR/logs/aegis.log" ]]; then
        local error_count=$(grep -c "ERROR\|CRITICAL" "$AEGIS_DIR/logs/aegis.log" 2>/dev/null || echo "0")
        log_info "Errors in AEGIS log: $error_count"
        
        if [[ $error_count -gt 10 ]]; then
            log_warning "High error count in AEGIS log: $error_count"
            if [[ "$SEND_ALERTS" == true ]]; then
                send_alert "High error count in AEGIS log: $error_count"
            fi
        fi
    fi
}

# Send alert notification
send_alert() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Send to Slack if webhook is configured
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 AEGIS Alert [$timestamp]: $message\"}" \
            "$SLACK_WEBHOOK_URL" >/dev/null 2>&1
    fi
    
    # Send email if configured
    if [[ -n "${ALERT_EMAIL:-}" ]] && command_exists mail; then
        echo "AEGIS Alert [$timestamp]: $message" | mail -s "AEGIS Alert" "$ALERT_EMAIL"
    fi
    
    # Write to alert log
    echo "[$timestamp] ALERT: $message" >> "$AEGIS_DIR/logs/alerts.log"
}

# Generate monitoring report
generate_report() {
    local report_file="$AEGIS_DIR/logs/monitor_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "monitoring_interval": $MONITOR_INTERVAL,
    "checks_performed": {
        "services": $CHECK_SERVICES,
        "resources": $CHECK_RESOURCES,
        "network": $CHECK_NETWORK,
        "security": $CHECK_SECURITY
    },
    "system_info": {
        "hostname": "$(hostname)",
        "uptime": "$(uptime -p 2>/dev/null || uptime)",
        "load_average": "$(uptime | awk -F'load average:' '{print $2}')",
        "disk_usage": $(df -h / | tail -1 | awk '{print "{\"filesystem\":\"" $1 "\",\"size\":\"" $2 "\",\"used\":\"" $3 "\",\"available\":\"" $4 "\",\"usage\":\"" $5 "\"}"}'),
        "memory_usage": $(free -m | awk 'NR==2{printf "{\"total\":%s,\"used\":%s,\"free\":%s,\"usage\":%.2f}", $2,$3,$4,$3*100/$2}')
    }
}
EOF
    
    log_info "Monitoring report generated: $report_file"
}

# Cleanup old logs
cleanup_logs() {
    log_info "Cleaning up old logs..."
    
    # Remove logs older than retention period
    find "$AEGIS_DIR/logs" -name "*.log" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "$AEGIS_DIR/logs" -name "monitor_report_*.json" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    
    # Rotate large log files
    if [[ -f "$LOG_FILE" ]]; then
        local log_size=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo "0")
        local max_size_bytes=$((100 * 1024 * 1024))  # 100MB
        
        if [[ $log_size -gt $max_size_bytes ]]; then
            mv "$LOG_FILE" "${LOG_FILE}.$(date +%Y%m%d_%H%M%S)"
            touch "$LOG_FILE"
            log_info "Log file rotated due to size"
        fi
    fi
}

# Run single monitoring cycle
run_monitoring_cycle() {
    log_header "Running monitoring cycle..."
    
    check_docker_services
    check_health_endpoints
    check_system_resources
    check_network_connectivity
    check_security_metrics
    check_log_files
    
    if [[ "$VERBOSE" == true ]]; then
        generate_report
    fi
    
    cleanup_logs
    
    log_success "Monitoring cycle completed"
}

# Signal handlers for daemon mode
cleanup_daemon() {
    log_info "Stopping monitoring daemon..."
    exit 0
}

# Main function
main() {
    # Parse arguments
    parse_arguments "$@"
    
    # Initialize
    initialize_monitoring
    
    # Set up signal handlers for daemon mode
    if [[ "$DAEMON_MODE" == true ]]; then
        trap cleanup_daemon SIGTERM SIGINT
        
        log_header "Starting AEGIS monitoring daemon (interval: ${MONITOR_INTERVAL}s)"
        
        while true; do
            run_monitoring_cycle
            sleep "$MONITOR_INTERVAL"
        done
    else
        run_monitoring_cycle
    fi
}

# Run main function with all arguments
main "$@"