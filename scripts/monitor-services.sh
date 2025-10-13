#!/bin/bash

# ============================================================================
# AEGIS Framework - Script de Monitoreo de Servicios para Linux
# ============================================================================
# Descripción: Script para monitorear el estado de todos los servicios AEGIS
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

set -euo pipefail

# Configuración por defecto
INTERVAL=30
CONTINUOUS=false
ALERTS=false
LOG_TO_FILE=false
LOG_PATH="logs/monitor.log"
JSON_OUTPUT=false
DETAILED=false
VERBOSE=false

# Colores para output
declare -A COLORS=(
    ["RED"]='\033[0;31m'
    ["GREEN"]='\033[0;32m'
    ["YELLOW"]='\033[1;33m'
    ["BLUE"]='\033[0;34m'
    ["CYAN"]='\033[0;36m'
    ["MAGENTA"]='\033[0;35m'
    ["WHITE"]='\033[1;37m'
    ["NC"]='\033[0m'
)

# Configuración de servicios
declare -A SERVICES=(
    ["dashboard_name"]="AEGIS Dashboard"
    ["dashboard_port"]="8080"
    ["dashboard_health"]="http://localhost:8080/health"
    ["dashboard_process"]="python.*main.py.*start-dashboard"
    
    ["securechat_name"]="Secure Chat UI"
    ["securechat_port"]="3000"
    ["securechat_health"]="http://localhost:3000"
    ["securechat_process"]="node.*npm run dev"
    
    ["blockchain_name"]="Local Blockchain"
    ["blockchain_port"]="8545"
    ["blockchain_health"]="http://localhost:8545"
    ["blockchain_process"]="node.*npx hardhat node"
    
    ["tor_name"]="Tor Service"
    ["tor_port"]="9050"
    ["tor_process"]="tor.*-f"
)

function print_color() {
    local color=$1
    local message=$2
    echo -e "${COLORS[$color]}${message}${COLORS[NC]}"
}

function show_help() {
    print_color "CYAN" "🛡️  AEGIS Framework - Monitor de Servicios"
    print_color "CYAN" "==========================================="
    echo ""
    print_color "YELLOW" "DESCRIPCIÓN:"
    echo "  Monitorea el estado de todos los servicios AEGIS en tiempo real"
    echo ""
    print_color "YELLOW" "USO:"
    echo "  ./monitor-services.sh [OPCIONES]"
    echo ""
    print_color "YELLOW" "OPCIONES:"
    echo "  -i, --interval <segundos>    Intervalo entre verificaciones (default: 30)"
    echo "  -c, --continuous             Monitoreo continuo (Ctrl+C para salir)"
    echo "  -a, --alerts                 Mostrar alertas cuando servicios fallen"
    echo "  -l, --log-to-file            Guardar logs en archivo"
    echo "  -p, --log-path <ruta>        Ruta del archivo de log (default: logs/monitor.log)"
    echo "  -j, --json                   Salida en formato JSON"
    echo "  -d, --detailed               Información detallada de cada servicio"
    echo "  -v, --verbose                Salida detallada"
    echo "  -h, --help                   Mostrar esta ayuda"
    echo ""
    print_color "YELLOW" "EJEMPLOS:"
    echo "  ./monitor-services.sh                                    # Verificación única"
    echo "  ./monitor-services.sh -c                                 # Monitoreo continuo"
    echo "  ./monitor-services.sh -c -i 10                           # Cada 10 segundos"
    echo "  ./monitor-services.sh -a -l                              # Con alertas y logs"
    echo "  ./monitor-services.sh -j                                 # Salida JSON"
    echo "  ./monitor-services.sh -d                                 # Información detallada"
    echo ""
    exit 0
}

function parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--interval)
                INTERVAL="$2"
                shift 2
                ;;
            -c|--continuous)
                CONTINUOUS=true
                shift
                ;;
            -a|--alerts)
                ALERTS=true
                shift
                ;;
            -l|--log-to-file)
                LOG_TO_FILE=true
                shift
                ;;
            -p|--log-path)
                LOG_PATH="$2"
                shift 2
                ;;
            -j|--json)
                JSON_OUTPUT=true
                shift
                ;;
            -d|--detailed)
                DETAILED=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                ;;
            *)
                print_color "RED" "❌ Opción desconocida: $1"
                echo "Usa -h o --help para ver las opciones disponibles."
                exit 1
                ;;
        esac
    done
}

function write_log() {
    local message=$1
    local level=${2:-"INFO"}
    
    if [[ "$LOG_TO_FILE" == "true" ]]; then
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local log_entry="[$timestamp] [$level] $message"
        
        # Crear directorio si no existe
        local log_dir=$(dirname "$LOG_PATH")
        if [[ ! -d "$log_dir" ]]; then
            mkdir -p "$log_dir"
        fi
        
        echo "$log_entry" >> "$LOG_PATH"
    fi
}

function test_port() {
    local port=$1
    local host=${2:-"localhost"}
    local timeout=${3:-3}
    
    if command -v nc >/dev/null 2>&1; then
        nc -z -w"$timeout" "$host" "$port" >/dev/null 2>&1
    elif command -v timeout >/dev/null 2>&1; then
        timeout "$timeout" bash -c "echo >/dev/tcp/$host/$port" >/dev/null 2>&1
    else
        # Fallback usando telnet
        timeout "$timeout" telnet "$host" "$port" >/dev/null 2>&1
    fi
}

function test_http_endpoint() {
    local url=$1
    local timeout=${2:-5}
    
    if command -v curl >/dev/null 2>&1; then
        local response=$(curl -s -w "%{http_code},%{time_total}" --max-time "$timeout" "$url" 2>/dev/null || echo "000,0")
        local status_code=$(echo "$response" | tail -1 | cut -d',' -f1)
        local response_time=$(echo "$response" | tail -1 | cut -d',' -f2)
        
        if [[ "$status_code" =~ ^[2-3][0-9][0-9]$ ]]; then
            echo "success,$status_code,$response_time"
        else
            echo "failed,$status_code,0"
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget --timeout="$timeout" --tries=1 -q --spider "$url" 2>/dev/null; then
            echo "success,200,0"
        else
            echo "failed,000,0"
        fi
    else
        echo "failed,000,0"
    fi
}

function get_process_info() {
    local process_pattern=$1
    
    local pids=$(pgrep -f "$process_pattern" 2>/dev/null || true)
    
    if [[ -n "$pids" ]]; then
        local pid=$(echo "$pids" | head -1)
        local cpu_percent=""
        local memory_mb=""
        local start_time=""
        local command_line=""
        
        if command -v ps >/dev/null 2>&1; then
            # Obtener información del proceso
            local ps_info=$(ps -p "$pid" -o pid,pcpu,rss,lstart,cmd --no-headers 2>/dev/null || true)
            
            if [[ -n "$ps_info" ]]; then
                cpu_percent=$(echo "$ps_info" | awk '{print $2}')
                memory_mb=$(echo "$ps_info" | awk '{printf "%.2f", $3/1024}')
                start_time=$(echo "$ps_info" | awk '{print $4" "$5" "$6" "$7" "$8}')
                command_line=$(echo "$ps_info" | cut -d' ' -f9-)
            fi
        fi
        
        echo "found,$pid,$cpu_percent,$memory_mb,$start_time,$command_line"
    else
        echo "not_found,,,,,"
    fi
}

function get_service_status() {
    local service_prefix=$1
    
    local name="${SERVICES[${service_prefix}_name]}"
    local port="${SERVICES[${service_prefix}_port]}"
    local health_endpoint="${SERVICES[${service_prefix}_health]:-}"
    local process_pattern="${SERVICES[${service_prefix}_process]}"
    
    local status="Unknown"
    local port_open=false
    local process_found=false
    local health_success=false
    local health_status_code=""
    local health_response_time=""
    local process_info=""
    
    # Verificar puerto
    if [[ -n "$port" ]]; then
        if test_port "$port"; then
            port_open=true
        fi
    fi
    
    # Verificar proceso
    if [[ -n "$process_pattern" ]]; then
        process_info=$(get_process_info "$process_pattern")
        if [[ "$process_info" == found,* ]]; then
            process_found=true
        fi
    fi
    
    # Verificar endpoint de salud
    if [[ -n "$health_endpoint" ]]; then
        local health_result=$(test_http_endpoint "$health_endpoint")
        local health_status=$(echo "$health_result" | cut -d',' -f1)
        health_status_code=$(echo "$health_result" | cut -d',' -f2)
        health_response_time=$(echo "$health_result" | cut -d',' -f3)
        
        if [[ "$health_status" == "success" ]]; then
            health_success=true
        fi
    fi
    
    # Determinar estado general
    local is_running=false
    
    if [[ -n "$health_endpoint" ]]; then
        is_running=$health_success
    elif [[ -n "$port" ]]; then
        is_running=$port_open
    elif [[ -n "$process_pattern" ]]; then
        is_running=$process_found
    fi
    
    if [[ "$is_running" == "true" ]]; then
        status="Running"
    else
        status="Stopped"
    fi
    
    # Crear JSON de respuesta
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat << EOF
{
    "name": "$name",
    "key": "$service_prefix",
    "status": "$status",
    "timestamp": "$timestamp",
    "port": {
        "number": "$port",
        "open": $port_open
    },
    "process": {
        "found": $process_found,
        "info": "$process_info"
    },
    "health": {
        "success": $health_success,
        "status_code": "$health_status_code",
        "response_time": "$health_response_time"
    }
}
EOF
}

function show_service_status() {
    local service_json=$1
    local show_detailed=${2:-false}
    
    local name=$(echo "$service_json" | jq -r '.name')
    local status=$(echo "$service_json" | jq -r '.status')
    local port_number=$(echo "$service_json" | jq -r '.port.number')
    local port_open=$(echo "$service_json" | jq -r '.port.open')
    local process_found=$(echo "$service_json" | jq -r '.process.found')
    local process_info=$(echo "$service_json" | jq -r '.process.info')
    local health_success=$(echo "$service_json" | jq -r '.health.success')
    local health_status_code=$(echo "$service_json" | jq -r '.health.status_code')
    
    local status_color="RED"
    local status_icon="❌"
    
    if [[ "$status" == "Running" ]]; then
        status_color="GREEN"
        status_icon="✅"
    fi
    
    print_color "$status_color" "$status_icon $name: $status"
    
    if [[ "$show_detailed" == "true" ]]; then
        # Información del puerto
        if [[ "$port_number" != "null" && "$port_number" != "" ]]; then
            local port_status="Cerrado"
            local port_color="RED"
            if [[ "$port_open" == "true" ]]; then
                port_status="Abierto"
                port_color="GREEN"
            fi
            print_color "$port_color" "   🔌 Puerto $port_number: $port_status"
        fi
        
        # Información del proceso
        if [[ "$process_found" == "true" && "$process_info" != "null" ]]; then
            IFS=',' read -r found pid cpu_percent memory_mb start_time command_line <<< "$process_info"
            print_color "WHITE" "   🔄 PID: $pid"
            if [[ -n "$memory_mb" ]]; then
                print_color "WHITE" "   💾 Memoria: ${memory_mb} MB"
            fi
            if [[ -n "$cpu_percent" ]]; then
                print_color "WHITE" "   🖥️  CPU: ${cpu_percent}%"
            fi
            if [[ -n "$start_time" ]]; then
                print_color "WHITE" "   ⏱️  Iniciado: $start_time"
            fi
        fi
        
        # Información de salud
        if [[ "$health_success" != "null" ]]; then
            if [[ "$health_success" == "true" ]]; then
                print_color "GREEN" "   🏥 Health Check: OK (HTTP $health_status_code)"
            else
                print_color "RED" "   🏥 Health Check: FAIL"
            fi
        fi
        
        echo ""
    fi
}

function show_system_summary() {
    local service_statuses=$1
    
    local running_count=$(echo "$service_statuses" | jq '[.[] | select(.status == "Running")] | length')
    local total_count=$(echo "$service_statuses" | jq 'length')
    
    echo ""
    print_color "CYAN" "📊 Resumen del Sistema:"
    print_color "CYAN" "======================"
    
    local summary_color="RED"
    if [[ "$running_count" -eq "$total_count" ]]; then
        summary_color="GREEN"
    elif [[ "$running_count" -gt 0 ]]; then
        summary_color="YELLOW"
    fi
    
    print_color "$summary_color" "🔧 Servicios activos: $running_count/$total_count"
    
    # Mostrar métricas del sistema
    if command -v top >/dev/null 2>&1; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' 2>/dev/null || echo "N/A")
        print_color "WHITE" "💻 CPU: $cpu_usage"
    fi
    
    if command -v free >/dev/null 2>&1; then
        local memory_info=$(free -h | grep "Mem:" | awk '{printf "%.1f GB / %.1f GB (%.1f%%)", $3, $2, ($3/$2)*100}' 2>/dev/null || echo "N/A")
        print_color "WHITE" "🧠 Memoria: $memory_info"
    fi
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    print_color "WHITE" "🕒 Última verificación: $timestamp"
}

function show_json_output() {
    local service_statuses=$1
    
    local running_count=$(echo "$service_statuses" | jq '[.[] | select(.status == "Running")] | length')
    local stopped_count=$(echo "$service_statuses" | jq '[.[] | select(.status == "Stopped")] | length')
    local total_count=$(echo "$service_statuses" | jq 'length')
    
    local timestamp=$(date '+%Y-%m-%dT%H:%M:%S')
    
    jq -n \
        --argjson services "$service_statuses" \
        --arg timestamp "$timestamp" \
        --argjson total "$total_count" \
        --argjson running "$running_count" \
        --argjson stopped "$stopped_count" \
        '{
            timestamp: $timestamp,
            services: $services,
            summary: {
                total: $total,
                running: $running,
                stopped: $stopped
            }
        }'
}

function send_alert() {
    local service_name=$1
    local status=$2
    local previous_status=$3
    
    if [[ "$status" != "$previous_status" ]]; then
        local alert_message="🚨 ALERTA: $service_name cambió de $previous_status a $status"
        
        if [[ "$status" == "Stopped" ]]; then
            print_color "RED" "$alert_message"
        elif [[ "$status" == "Running" && "$previous_status" == "Stopped" ]]; then
            print_color "GREEN" "✅ RECUPERADO: $service_name está funcionando nuevamente"
        fi
        
        write_log "$alert_message" "ALERT"
    fi
}

function start_monitoring() {
    print_color "CYAN" "🛡️  Iniciando monitoreo de servicios AEGIS..."
    print_color "WHITE" "Intervalo: $INTERVAL segundos"
    print_color "YELLOW" "Presiona Ctrl+C para detener"
    echo ""
    
    declare -A previous_statuses
    
    # Configurar manejo de señales
    trap 'print_color "YELLOW" "\n🛑 Monitoreo detenido por el usuario"; write_log "Monitoreo detenido por el usuario" "INFO"; exit 0' INT TERM
    
    while true; do
        if [[ "$JSON_OUTPUT" != "true" ]]; then
            clear
            print_color "CYAN" "🛡️  AEGIS Framework - Monitor de Servicios"
            print_color "CYAN" "==========================================="
            echo ""
        fi
        
        local service_statuses="[]"
        
        for service_key in dashboard securechat blockchain tor; do
            local status_json=$(get_service_status "$service_key")
            service_statuses=$(echo "$service_statuses" | jq ". += [$status_json]")
            
            local current_status=$(echo "$status_json" | jq -r '.status')
            local service_name=$(echo "$status_json" | jq -r '.name')
            
            # Enviar alertas si está habilitado
            if [[ "$ALERTS" == "true" && -n "${previous_statuses[$service_key]:-}" ]]; then
                send_alert "$service_name" "$current_status" "${previous_statuses[$service_key]}"
            fi
            
            previous_statuses[$service_key]="$current_status"
            
            # Mostrar estado del servicio
            if [[ "$JSON_OUTPUT" != "true" ]]; then
                show_service_status "$status_json" "$DETAILED"
            fi
            
            # Log del estado
            write_log "$service_name: $current_status"
        done
        
        if [[ "$JSON_OUTPUT" == "true" ]]; then
            show_json_output "$service_statuses"
        else
            show_system_summary "$service_statuses"
        fi
        
        if [[ "$CONTINUOUS" != "true" ]]; then
            break
        fi
        
        sleep "$INTERVAL"
    done
}

function check_prerequisites() {
    # Verificar si estamos en el directorio correcto
    if [[ ! -f "main.py" ]]; then
        print_color "RED" "❌ Error: No se encontró main.py. Ejecuta este script desde el directorio raíz del proyecto AEGIS."
        exit 1
    fi
    
    # Verificar herramientas necesarias
    local missing_tools=()
    
    if ! command -v jq >/dev/null 2>&1; then
        missing_tools+=("jq")
    fi
    
    if ! command -v pgrep >/dev/null 2>&1; then
        missing_tools+=("pgrep (procps)")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_color "RED" "❌ Herramientas faltantes: ${missing_tools[*]}"
        print_color "YELLOW" "Instala las herramientas faltantes:"
        print_color "WHITE" "  Ubuntu/Debian: sudo apt-get install jq procps"
        print_color "WHITE" "  CentOS/RHEL:   sudo yum install jq procps-ng"
        print_color "WHITE" "  Arch Linux:    sudo pacman -S jq procps-ng"
        exit 1
    fi
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function main() {
    parse_arguments "$@"
    
    if [[ "$VERBOSE" == "true" ]]; then
        set -x
    fi
    
    check_prerequisites
    
    try {
        if [[ "$CONTINUOUS" == "true" ]]; then
            start_monitoring
        else
            # Verificación única
            print_color "CYAN" "🛡️  AEGIS Framework - Estado de Servicios"
            print_color "CYAN" "========================================="
            echo ""
            
            local service_statuses="[]"
            
            for service_key in dashboard securechat blockchain tor; do
                local status_json=$(get_service_status "$service_key")
                service_statuses=$(echo "$service_statuses" | jq ". += [$status_json]")
                
                local service_name=$(echo "$status_json" | jq -r '.name')
                local current_status=$(echo "$status_json" | jq -r '.status')
                
                if [[ "$JSON_OUTPUT" != "true" ]]; then
                    show_service_status "$status_json" "$DETAILED"
                fi
                
                write_log "$service_name: $current_status"
            done
            
            if [[ "$JSON_OUTPUT" == "true" ]]; then
                show_json_output "$service_statuses"
            else
                show_system_summary "$service_statuses"
            fi
        fi
    } || {
        print_color "RED" "❌ Error durante el monitoreo: $?"
        write_log "Error durante el monitoreo" "ERROR"
        exit 1
    }
}

# Ejecutar función principal
main "$@"