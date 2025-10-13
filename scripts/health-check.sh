#!/bin/bash
# ============================================================================
# AEGIS Framework - Verificador de Salud del Sistema para Linux
# ============================================================================
# Descripción: Script para verificar el estado de salud de todos los servicios
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

set -euo pipefail

# Configuración de colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Variables globales
DETAILED=false
JSON_OUTPUT=false
CONTINUOUS=false
INTERVAL=30
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Funciones de utilidad
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_error() {
    print_color "$RED" "❌ ERROR: $1"
}

print_success() {
    print_color "$GREEN" "✅ $1"
}

print_warning() {
    print_color "$YELLOW" "⚠️  WARNING: $1"
}

print_info() {
    print_color "$BLUE" "ℹ️  $1"
}

show_help() {
    print_color "$CYAN" "🛡️  AEGIS Framework - Verificador de Salud del Sistema"
    print_color "$CYAN" "======================================================="
    echo ""
    print_color "$YELLOW" "DESCRIPCIÓN:"
    echo "  Verifica el estado de salud de todos los servicios AEGIS"
    echo ""
    print_color "$YELLOW" "USO:"
    echo "  ./health-check.sh [OPCIONES]"
    echo ""
    print_color "$YELLOW" "OPCIONES:"
    echo "  -d, --detailed     Mostrar información detallada"
    echo "  -j, --json         Salida en formato JSON"
    echo "  -c, --continuous   Monitoreo continuo"
    echo "  -i, --interval N   Intervalo en segundos para monitoreo continuo (default: 30)"
    echo "  -h, --help         Mostrar esta ayuda"
    echo ""
    print_color "$YELLOW" "EJEMPLOS:"
    echo "  ./health-check.sh"
    echo "  ./health-check.sh --detailed"
    echo "  ./health-check.sh --json"
    echo "  ./health-check.sh --continuous --interval 60"
    echo ""
    exit 0
}

# Parsear argumentos
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--detailed)
                DETAILED=true
                shift
                ;;
            -j|--json)
                JSON_OUTPUT=true
                shift
                ;;
            -c|--continuous)
                CONTINUOUS=true
                shift
                ;;
            -i|--interval)
                INTERVAL="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                ;;
            *)
                print_error "Opción desconocida: $1"
                show_help
                ;;
        esac
    done
}

# Verificar requisitos del sistema
test_system_requirements() {
    local results=""
    
    # Verificar OS
    local os_info=""
    local os_status="Unknown"
    local os_healthy=false
    
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        os_info="$NAME $VERSION"
        os_status="OK"
        os_healthy=true
    else
        os_info="Unknown Linux distribution"
        os_status="Unknown"
    fi
    
    # Verificar kernel
    local kernel_version=$(uname -r)
    local kernel_status="OK"
    local kernel_healthy=true
    
    # Verificar memoria
    local memory_info=""
    local memory_status="Unknown"
    local memory_healthy=false
    
    if command -v free >/dev/null 2>&1; then
        local total_memory_kb=$(free | grep '^Mem:' | awk '{print $2}')
        local total_memory_gb=$((total_memory_kb / 1024 / 1024))
        memory_info="${total_memory_gb} GB"
        
        if [[ $total_memory_gb -ge 4 ]]; then
            memory_status="OK"
            memory_healthy=true
        else
            memory_status="Insufficient"
        fi
    else
        memory_info="Cannot determine"
        memory_status="Error"
    fi
    
    # Verificar espacio en disco
    local disk_info=""
    local disk_status="Unknown"
    local disk_healthy=false
    
    if command -v df >/dev/null 2>&1; then
        local free_space_gb=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
        disk_info="${free_space_gb} GB free"
        
        if [[ $free_space_gb -ge 5 ]]; then
            disk_status="OK"
            disk_healthy=true
        else
            disk_status="Low Space"
        fi
    else
        disk_info="Cannot determine"
        disk_status="Error"
    fi
    
    # Verificar bash
    local bash_version=$BASH_VERSION
    local bash_status="OK"
    local bash_healthy=true
    
    if $JSON_OUTPUT; then
        cat << EOF
{
  "OS": {
    "Status": "$os_status",
    "Details": "$os_info",
    "Healthy": $os_healthy
  },
  "Kernel": {
    "Status": "$kernel_status",
    "Details": "$kernel_version",
    "Healthy": $kernel_healthy
  },
  "Memory": {
    "Status": "$memory_status",
    "Details": "$memory_info",
    "Healthy": $memory_healthy
  },
  "Disk": {
    "Status": "$disk_status",
    "Details": "$disk_info",
    "Healthy": $disk_healthy
  },
  "Bash": {
    "Status": "$bash_status",
    "Details": "$bash_version",
    "Healthy": $bash_healthy
  }
}
EOF
    else
        echo "OS:$os_status:$os_info:$os_healthy"
        echo "Kernel:$kernel_status:$kernel_version:$kernel_healthy"
        echo "Memory:$memory_status:$memory_info:$memory_healthy"
        echo "Disk:$disk_status:$disk_info:$disk_healthy"
        echo "Bash:$bash_status:$bash_version:$bash_healthy"
    fi
}

# Verificar dependencias
test_dependencies() {
    local python_status="Unknown"
    local python_details=""
    local python_healthy=false
    
    local node_status="Unknown"
    local node_details=""
    local node_healthy=false
    
    local npm_status="Unknown"
    local npm_details=""
    local npm_healthy=false
    
    local git_status="Unknown"
    local git_details=""
    local git_healthy=false
    
    local tor_status="Unknown"
    local tor_details=""
    local tor_healthy=false
    
    # Verificar Python
    if command -v python3 >/dev/null 2>&1; then
        python_details=$(python3 --version 2>&1)
        if echo "$python_details" | grep -qE "Python 3\.(8|9|1[0-9])"; then
            python_status="OK"
            python_healthy=true
        else
            python_status="Incompatible Version"
        fi
    else
        python_status="Not Found"
        python_details="Python3 no está instalado o no está en PATH"
    fi
    
    # Verificar Node.js
    if command -v node >/dev/null 2>&1; then
        node_details=$(node --version 2>&1)
        if echo "$node_details" | grep -qE "v(1[8-9]|2[0-9])"; then
            node_status="OK"
            node_healthy=true
        else
            node_status="Incompatible Version"
        fi
    else
        node_status="Not Found"
        node_details="Node.js no está instalado o no está en PATH"
    fi
    
    # Verificar npm
    if command -v npm >/dev/null 2>&1; then
        npm_details="v$(npm --version 2>&1)"
        npm_status="OK"
        npm_healthy=true
    else
        npm_status="Not Found"
        npm_details="npm no está instalado o no está en PATH"
    fi
    
    # Verificar Git
    if command -v git >/dev/null 2>&1; then
        git_details=$(git --version 2>&1)
        git_status="OK"
        git_healthy=true
    else
        git_status="Not Found"
        git_details="Git no está instalado o no está en PATH"
    fi
    
    # Verificar Tor
    if command -v tor >/dev/null 2>&1; then
        tor_details=$(tor --version 2>&1 | head -1)
        tor_status="OK"
        tor_healthy=true
    else
        tor_status="Not Found"
        tor_details="Tor no está instalado o no está en PATH"
    fi
    
    if $JSON_OUTPUT; then
        cat << EOF
{
  "Python": {
    "Status": "$python_status",
    "Details": "$python_details",
    "Healthy": $python_healthy
  },
  "Node.js": {
    "Status": "$node_status",
    "Details": "$node_details",
    "Healthy": $node_healthy
  },
  "npm": {
    "Status": "$npm_status",
    "Details": "$npm_details",
    "Healthy": $npm_healthy
  },
  "Git": {
    "Status": "$git_status",
    "Details": "$git_details",
    "Healthy": $git_healthy
  },
  "Tor": {
    "Status": "$tor_status",
    "Details": "$tor_details",
    "Healthy": $tor_healthy
  }
}
EOF
    else
        echo "Python:$python_status:$python_details:$python_healthy"
        echo "Node.js:$node_status:$node_details:$node_healthy"
        echo "npm:$npm_status:$npm_details:$npm_healthy"
        echo "Git:$git_status:$git_details:$git_healthy"
        echo "Tor:$tor_status:$tor_details:$tor_healthy"
    fi
}

# Verificar estructura del proyecto
test_project_structure() {
    cd "$PROJECT_ROOT"
    
    # Verificar archivos de configuración
    local config_files=(".env" "config/app_config.json" "config/torrc")
    local missing_configs=()
    local existing_configs=()
    
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            existing_configs+=("$file")
        else
            missing_configs+=("$file")
        fi
    done
    
    local config_status="OK"
    local config_healthy=true
    if [[ ${#missing_configs[@]} -gt 0 ]]; then
        config_status="Missing Files"
        config_healthy=false
    fi
    
    # Verificar directorios
    local required_dirs=("config" "logs" "tor_data" "dapps/secure-chat/ui" "dapps/aegis-token")
    local missing_dirs=()
    local existing_dirs=()
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            existing_dirs+=("$dir")
        else
            missing_dirs+=("$dir")
        fi
    done
    
    local dirs_status="OK"
    local dirs_healthy=true
    if [[ ${#missing_dirs[@]} -gt 0 ]]; then
        dirs_status="Missing Directories"
        dirs_healthy=false
    fi
    
    # Verificar entorno virtual de Python
    local venv_status="Not Found"
    local venv_details="Python virtual environment not found"
    local venv_healthy=false
    
    if [[ -f "venv/bin/activate" ]]; then
        venv_status="OK"
        venv_details="Virtual environment found"
        venv_healthy=true
    fi
    
    # Verificar node_modules
    local node_modules_paths=("dapps/secure-chat/ui/node_modules" "dapps/aegis-token/node_modules")
    local missing_node_modules=()
    local existing_node_modules=()
    
    for path in "${node_modules_paths[@]}"; do
        if [[ -d "$path" ]]; then
            existing_node_modules+=("$path")
        else
            missing_node_modules+=("$path")
        fi
    done
    
    local node_modules_status="OK"
    local node_modules_healthy=true
    if [[ ${#missing_node_modules[@]} -gt 0 ]]; then
        node_modules_status="Missing Dependencies"
        node_modules_healthy=false
    fi
    
    if $JSON_OUTPUT; then
        cat << EOF
{
  "ConfigFiles": {
    "Status": "$config_status",
    "Details": {
      "Existing": [$(printf '"%s",' "${existing_configs[@]}" | sed 's/,$//')]",
      "Missing": [$(printf '"%s",' "${missing_configs[@]}" | sed 's/,$//')]"
    },
    "Healthy": $config_healthy
  },
  "Directories": {
    "Status": "$dirs_status",
    "Details": {
      "Existing": [$(printf '"%s",' "${existing_dirs[@]}" | sed 's/,$//')]",
      "Missing": [$(printf '"%s",' "${missing_dirs[@]}" | sed 's/,$//')]"
    },
    "Healthy": $dirs_healthy
  },
  "VirtualEnv": {
    "Status": "$venv_status",
    "Details": "$venv_details",
    "Healthy": $venv_healthy
  },
  "NodeModules": {
    "Status": "$node_modules_status",
    "Details": {
      "Existing": [$(printf '"%s",' "${existing_node_modules[@]}" | sed 's/,$//')]",
      "Missing": [$(printf '"%s",' "${missing_node_modules[@]}" | sed 's/,$//')]"
    },
    "Healthy": $node_modules_healthy
  }
}
EOF
    else
        echo "ConfigFiles:$config_status:${existing_configs[*]}|${missing_configs[*]}:$config_healthy"
        echo "Directories:$dirs_status:${existing_dirs[*]}|${missing_dirs[*]}:$dirs_healthy"
        echo "VirtualEnv:$venv_status:$venv_details:$venv_healthy"
        echo "NodeModules:$node_modules_status:${existing_node_modules[*]}|${missing_node_modules[*]}:$node_modules_healthy"
    fi
}

# Verificar puertos de red
test_network_ports() {
    local ports=(
        "Dashboard:8080"
        "SecureChat:5173"
        "Blockchain:8545"
        "TorSOCKS:9050"
        "TorControl:9051"
    )
    
    local results=()
    
    for port_info in "${ports[@]}"; do
        local service_name="${port_info%:*}"
        local port="${port_info#*:}"
        
        local status="Closed"
        local details="Service is not running"
        local healthy=false
        
        if command -v netstat >/dev/null 2>&1; then
            if netstat -tuln 2>/dev/null | grep -q ":$port "; then
                status="Open"
                details="Service is running"
                healthy=true
            fi
        elif command -v ss >/dev/null 2>&1; then
            if ss -tuln 2>/dev/null | grep -q ":$port "; then
                status="Open"
                details="Service is running"
                healthy=true
            fi
        else
            status="Error"
            details="Cannot check port status"
        fi
        
        if $JSON_OUTPUT; then
            results+=("\"$service_name\": {\"Port\": $port, \"Status\": \"$status\", \"Details\": \"$details\", \"Healthy\": $healthy}")
        else
            echo "$service_name:$status:$details:$healthy"
        fi
    done
    
    if $JSON_OUTPUT; then
        echo "{"
        printf "%s,\n" "${results[@]}" | sed '$s/,$//'
        echo "}"
    fi
}

# Verificar salud de servicios
test_service_health() {
    local services=(
        "Dashboard:http://localhost:8080"
        "SecureChat:http://localhost:5173"
        "Blockchain:http://localhost:8545"
    )
    
    local results=()
    
    for service_info in "${services[@]}"; do
        local service_name="${service_info%:*}"
        local url="${service_info#*:}"
        
        local status="Unreachable"
        local details="Service is not responding"
        local healthy=false
        
        if command -v curl >/dev/null 2>&1; then
            local response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$url" 2>/dev/null || echo "000")
            
            if [[ "$response_code" == "200" ]]; then
                status="Healthy"
                details="HTTP 200 OK"
                healthy=true
            elif [[ "$response_code" != "000" ]]; then
                status="Unhealthy"
                details="HTTP $response_code"
            else
                status="Unreachable"
                details="Connection failed"
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget --spider --timeout=5 "$url" >/dev/null 2>&1; then
                status="Healthy"
                details="Service responding"
                healthy=true
            fi
        else
            status="Error"
            details="Cannot test service health"
        fi
        
        if $JSON_OUTPUT; then
            results+=("\"$service_name\": {\"URL\": \"$url\", \"Status\": \"$status\", \"Details\": \"$details\", \"Healthy\": $healthy}")
        else
            echo "$service_name:$status:$details:$healthy"
        fi
    done
    
    if $JSON_OUTPUT; then
        echo "{"
        printf "%s,\n" "${results[@]}" | sed '$s/,$//'
        echo "}"
    fi
}

# Obtener métricas del sistema
get_system_metrics() {
    local cpu_usage=0
    local cpu_details="Unknown"
    
    local memory_usage=0
    local memory_details="Unknown"
    
    local disk_usage=0
    local disk_details="Unknown"
    
    local network_status="Unknown"
    local network_details="Unknown"
    
    # CPU Usage
    if command -v top >/dev/null 2>&1; then
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' 2>/dev/null || echo "0")
        cpu_details="${cpu_usage}% usage"
    fi
    
    # Memory Usage
    if command -v free >/dev/null 2>&1; then
        local mem_info=$(free | grep '^Mem:')
        local total_mem=$(echo "$mem_info" | awk '{print $2}')
        local used_mem=$(echo "$mem_info" | awk '{print $3}')
        
        if [[ $total_mem -gt 0 ]]; then
            memory_usage=$(( (used_mem * 100) / total_mem ))
            memory_details="${memory_usage}% used ($((used_mem / 1024 / 1024)) GB / $((total_mem / 1024 / 1024)) GB)"
        fi
    fi
    
    # Disk Usage
    if command -v df >/dev/null 2>&1; then
        local disk_info=$(df -h . | tail -1)
        disk_usage=$(echo "$disk_info" | awk '{print $5}' | sed 's/%//')
        local used_space=$(echo "$disk_info" | awk '{print $3}')
        local total_space=$(echo "$disk_info" | awk '{print $2}')
        disk_details="${disk_usage}% used ($used_space / $total_space)"
    fi
    
    # Network connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        network_status="Connected"
        network_details="Internet connectivity available"
    else
        network_status="Disconnected"
        network_details="No internet connectivity"
    fi
    
    if $JSON_OUTPUT; then
        cat << EOF
{
  "CPU": {
    "Usage": $cpu_usage,
    "Details": "$cpu_details"
  },
  "Memory": {
    "Usage": $memory_usage,
    "Details": "$memory_details"
  },
  "Disk": {
    "Usage": $disk_usage,
    "Details": "$disk_details"
  },
  "Network": {
    "Status": "$network_status",
    "Details": "$network_details"
  }
}
EOF
    else
        echo "CPU:$cpu_usage:$cpu_details"
        echo "Memory:$memory_usage:$memory_details"
        echo "Disk:$disk_usage:$disk_details"
        echo "Network:$network_status:$network_details"
    fi
}

# Mostrar reporte de salud
show_health_report() {
    if $JSON_OUTPUT; then
        echo "{"
        echo "  \"SystemRequirements\": $(test_system_requirements),"
        echo "  \"Dependencies\": $(test_dependencies),"
        echo "  \"ProjectStructure\": $(test_project_structure),"
        echo "  \"NetworkPorts\": $(test_network_ports),"
        echo "  \"ServiceHealth\": $(test_service_health)"
        
        if $DETAILED; then
            echo "  ,\"Metrics\": $(get_system_metrics)"
        fi
        
        echo "}"
        return
    fi
    
    print_color "$CYAN" "🛡️  AEGIS Framework - Reporte de Salud del Sistema"
    print_color "$CYAN" "================================================="
    echo ""
    
    # Recopilar datos
    local system_data=$(test_system_requirements)
    local deps_data=$(test_dependencies)
    local structure_data=$(test_project_structure)
    local ports_data=$(test_network_ports)
    local services_data=$(test_service_health)
    
    # Calcular resumen
    local total_checks=0
    local healthy_checks=0
    
    # Contar verificaciones saludables
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            total_checks=$((total_checks + 1))
            if [[ "$line" == *":true" ]]; then
                healthy_checks=$((healthy_checks + 1))
            fi
        fi
    done <<< "$system_data"$'\n'"$deps_data"$'\n'"$structure_data"$'\n'"$ports_data"$'\n'"$services_data"
    
    local health_percentage=0
    if [[ $total_checks -gt 0 ]]; then
        health_percentage=$(( (healthy_checks * 100) / total_checks ))
    fi
    
    print_color "$YELLOW" "📊 RESUMEN GENERAL"
    echo "Verificaciones saludables: $healthy_checks/$total_checks ($health_percentage%)"
    echo ""
    
    # Mostrar cada categoría
    show_category "🔍 Requisitos del Sistema" "$system_data"
    show_category "🔍 Dependencias" "$deps_data"
    show_category "🔍 Estructura del Proyecto" "$structure_data"
    show_category "🔍 Puertos de Red" "$ports_data"
    show_category "🔍 Salud de Servicios" "$services_data"
    
    if $DETAILED; then
        local metrics_data=$(get_system_metrics)
        show_metrics "📈 Métricas del Sistema" "$metrics_data"
    fi
    
    # Mostrar recomendaciones
    if [[ $health_percentage -lt 100 ]]; then
        show_recommendations "$system_data" "$deps_data" "$structure_data" "$ports_data" "$services_data"
    fi
    
    # Estado general
    if [[ $health_percentage -eq 100 ]]; then
        print_success "¡Todos los sistemas están funcionando correctamente!"
    elif [[ $health_percentage -ge 80 ]]; then
        print_warning "La mayoría de los sistemas están funcionando, pero hay algunos problemas menores"
    else
        print_error "Se detectaron problemas significativos que requieren atención"
    fi
    
    echo ""
    print_color "$CYAN" "🕒 Última verificación: $(date '+%Y-%m-%d %H:%M:%S')"
}

# Mostrar categoría de verificaciones
show_category() {
    local title="$1"
    local data="$2"
    
    print_color "$BLUE" "$title"
    print_color "$BLUE" "$(echo "$title" | sed 's/./=/g')"
    
    while IFS=':' read -r name status details healthy; do
        if [[ -n "$name" ]]; then
            local icon="❌"
            local color="$RED"
            
            if [[ "$healthy" == "true" ]]; then
                icon="✅"
                color="$GREEN"
            fi
            
            echo -e "$icon $name: ${color}$status${NC}"
            
            if $DETAILED && [[ -n "$details" ]]; then
                echo "   $details"
            fi
        fi
    done <<< "$data"
    
    echo ""
}

# Mostrar métricas
show_metrics() {
    local title="$1"
    local data="$2"
    
    print_color "$MAGENTA" "$title"
    print_color "$MAGENTA" "$(echo "$title" | sed 's/./=/g')"
    
    while IFS=':' read -r metric usage details; do
        if [[ -n "$metric" ]]; then
            local color="$GREEN"
            
            if [[ $usage -gt 80 ]]; then
                color="$RED"
            elif [[ $usage -gt 60 ]]; then
                color="$YELLOW"
            fi
            
            echo -e "$metric: ${color}$details${NC}"
        fi
    done <<< "$data"
    
    echo ""
}

# Mostrar recomendaciones
show_recommendations() {
    print_color "$YELLOW" "💡 RECOMENDACIONES"
    print_color "$YELLOW" "=================="
    
    local all_data="$1"$'\n'"$2"$'\n'"$3"$'\n'"$4"$'\n'"$5"
    
    while IFS=':' read -r name status details healthy; do
        if [[ -n "$name" && "$healthy" == "false" ]]; then
            case "$status" in
                "Not Found")
                    echo "• Instala $name usando el script de dependencias"
                    ;;
                "Incompatible Version")
                    echo "• Actualiza $name a una versión compatible"
                    ;;
                "Missing Files")
                    echo "• Ejecuta el script de configuración para crear archivos faltantes"
                    ;;
                "Missing Directories")
                    echo "• Crea los directorios faltantes o ejecuta el script de configuración"
                    ;;
                "Closed")
                    echo "• Inicia el servicio $name"
                    ;;
                "Unreachable")
                    echo "• Verifica que el servicio $name esté ejecutándose correctamente"
                    ;;
            esac
        fi
    done <<< "$all_data"
    
    echo ""
}

# Monitoreo continuo
start_continuous_monitoring() {
    print_color "$BLUE" "🔄 Iniciando monitoreo continuo (intervalo: $INTERVAL segundos)"
    print_color "$YELLOW" "Presiona Ctrl+C para detener"
    echo ""
    
    trap 'print_color "$YELLOW" "Monitoreo detenido"; exit 0' INT
    
    while true; do
        clear
        show_health_report
        sleep "$INTERVAL"
    done
}

# Función principal
main() {
    parse_arguments "$@"
    
    if $CONTINUOUS; then
        start_continuous_monitoring
        return
    fi
    
    print_color "$BLUE" "🔍 Recopilando información del sistema..."
    echo ""
    
    show_health_report
}

# Ejecutar función principal
main "$@"