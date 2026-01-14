#!/bin/bash
# AEGIS Framework - Script de Inicio de Producción
# Script para iniciar todos los componentes en producción

set -e  # Salir en caso de error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables de configuración
AEGIS_HOME="${AEGIS_HOME:-$(pwd)}"
LOG_DIR="${AEGIS_HOME}/logs"
PID_DIR="${AEGIS_HOME}/pids"
CONFIG_FILE="${AEGIS_HOME}/production_config_v3.json"

# Funciones de utilidad
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

check_requirements() {
    log_info "Verificando requisitos del sistema..."
    
    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 no está instalado"
        exit 1
    fi
    
    # Verificar versión de Python
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ "$(printf '%s\n' "3.8" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.8" ]]; then
        log_error "Python 3.8+ requerido, versión actual: $PYTHON_VERSION"
        exit 1
    fi
    
    # Verificar módulos de Python
    local modules=("gunicorn" "uvicorn" "flask" "fastapi" "redis")
    for module in "${modules[@]}"; do
        if ! python3 -c "import $module" &> /dev/null; then
            log_error "Módulo Python faltante: $module"
            log_info "Instala con: pip3 install $module"
            exit 1
        fi
    done
    
    # Verificar espacio en disco
    local available_space=$(df "$AEGIS_HOME" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 1048576 ]]; then  # 1GB en KB
        log_warning "Espacio en disco bajo: $(($available_space / 1024))MB disponibles"
    fi
    
    log_success "Requisitos verificados"
}

create_directories() {
    log_info "Creando directorios necesarios..."
    
    mkdir -p "$LOG_DIR" "$PID_DIR"
    
    # Crear logs individuales para cada servicio
    touch "$LOG_DIR/node.log"
    touch "$LOG_DIR/api.log"
    touch "$LOG_DIR/dashboard.log"
    touch "$LOG_DIR/admin.log"
    touch "$LOG_DIR/deploy.log"
    
    log_success "Directorios creados"
}

check_config() {
    log_info "Verificando configuración..."
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Archivo de configuración no encontrado: $CONFIG_FILE"
        log_info "Crea production_config_v3.json con la configuración de producción"
        exit 1
    fi
    
    # Validar JSON
    if ! python3 -m json.tool "$CONFIG_FILE" &> /dev/null; then
        log_error "Archivo de configuración JSON inválido"
        exit 1
    fi
    
    log_success "Configuración válida"
}

start_service() {
    local service_name=$1
    local port=$2
    local workers=$3
    local module=$4
    local server_type=$5
    local log_file="$LOG_DIR/${service_name}.log"
    local pid_file="$PID_DIR/${service_name}.pid"
    
    log_info "Iniciando $service_name en puerto $port..."
    
    # Verificar si ya está corriendo
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_warning "$service_name ya está corriendo (PID: $pid)"
            return 0
        fi
    fi
    
    # Construir comando
    if [[ "$server_type" == "gunicorn" ]]; then
        local cmd=(
            python3 -m gunicorn
            --bind "127.0.0.1:$port"
            --workers "$workers"
            --worker-class sync
            --max-requests 1000
            --timeout 30
            --keepalive 2
            --log-level info
            --preload
            --daemon
            --pid "$pid_file"
            --access-logfile "$log_file"
            --error-logfile "$log_file"
            "$module"
        )
    else  # uvicorn
        local cmd=(
            python3 -m uvicorn
            "$module"
            --host "127.0.0.1"
            --port "$port"
            --workers "$workers"
            --log-level info
            --daemon
            --pid "$pid_file"
        )
    fi
    
    # Iniciar servicio
    if "${cmd[@]}" >> "$log_file" 2>&1; then
        sleep 2
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            log_success "$service_name iniciado (PID: $pid)"
            return 0
        else
            log_error "$service_name falló al iniciar"
            return 1
        fi
    else
        log_error "$service_name falló al iniciar"
        return 1
    fi
}

start_all_services() {
    log_info "Iniciando todos los servicios AEGIS..."
    
    # Leer configuración
    local node_port=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['node']['port'])" 2>/dev/null || echo "8080")
    local api_port=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['api']['port'])" 2>/dev/null || echo "8000")
    local dashboard_port=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['dashboard']['port'])" 2>/dev/null || echo "3000")
    local admin_port=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['admin']['port'])" 2>/dev/null || echo "8081")
    
    # Iniciar servicios
    start_service "aegis-node" "$node_port" 4 "node:app" "gunicorn"
    start_service "aegis-api" "$api_port" 4 "api:app" "gunicorn"
    start_service "aegis-dashboard" "$dashboard_port" 2 "dashboard:app" "uvicorn"
    start_service "aegis-admin" "$admin_port" 2 "admin:app" "uvicorn"
    
    log_success "Todos los servicios iniciados"
}

stop_service() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_info "Deteniendo $service_name (PID: $pid)..."
            kill -TERM "$pid"
            
            # Esperar hasta 10 segundos
            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [[ $count -lt 10 ]]; do
                sleep 1
                ((count++))
            done
            
            # Forzar si es necesario
            if ps -p "$pid" > /dev/null 2>&1; then
                log_warning "Forzando detención de $service_name"
                kill -KILL "$pid"
            fi
            
            rm -f "$pid_file"
            log_success "$service_name detenido"
        else
            log_warning "$service_name no está corriendo"
            rm -f "$pid_file"
        fi
    else
        log_warning "$service_name no está corriendo"
    fi
}

stop_all_services() {
    log_info "Deteniendo todos los servicios..."
    
    local services=("aegis-node" "aegis-api" "aegis-dashboard" "aegis-admin")
    for service in "${services[@]}"; do
        stop_service "$service"
    done
    
    log_success "Todos los servicios detenidos"
}

show_status() {
    log_info "Estado de servicios AEGIS:"
    
    local services=("aegis-node" "aegis-api" "aegis-dashboard" "aegis-admin")
    local running_count=0
    
    for service in "${services[@]}"; do
        local pid_file="$PID_DIR/${service}.pid"
        local status="🔴 DETENIDO"
        
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if ps -p "$pid" > /dev/null 2>&1; then
                status="🟢 CORRIENDO (PID: $pid)"
                ((running_count++))
            fi
        fi
        
        echo -e "  $status $service"
    done
    
    echo -e "\n📊 Resumen: $running_count/${#services[@]} servicios activos"
}

show_logs() {
    local service=$1
    local log_file="$LOG_DIR/${service}.log"
    
    if [[ -f "$log_file" ]]; then
        log_info "Mostrando logs de $service (últimas 50 líneas):"
        tail -n 50 "$log_file"
    else
        log_error "No se encontró log para $service"
    fi
}

show_help() {
    echo "Uso: $0 {start|stop|restart|status|logs|help}"
    echo ""
    echo "Comandos:"
    echo "  start   - Iniciar todos los servicios"
    echo "  stop    - Detener todos los servicios"
    echo "  restart - Reiniciar todos los servicios"
    echo "  status  - Mostrar estado de los servicios"
    echo "  logs    - Mostrar logs [servicio]"
    echo "  help    - Mostrar esta ayuda"
    echo ""
    echo "Servicios disponibles:"
    echo "  aegis-node, aegis-api, aegis-dashboard, aegis-admin"
    echo ""
    echo "Ejemplos:"
    echo "  $0 start"
    echo "  $0 logs aegis-node"
    echo "  $0 restart"
}

# Función principal
main() {
    case "${1:-help}" in
        start)
            check_requirements
            create_directories
            check_config
            start_all_services
            show_status
            log_success "Despliegue de producción completado"
            log_info "URLs de acceso:"
            log_info "  • Node: http://127.0.0.1:8080"
            log_info "  • API: http://127.0.0.1:8000"
            log_info "  • Dashboard: http://127.0.0.1:3000"
            log_info "  • Admin: http://127.0.0.1:8081"
            ;;
        stop)
            stop_all_services
            ;;
        restart)
            stop_all_services
            sleep 2
            check_requirements
            create_directories
            check_config
            start_all_services
            show_status
            ;;
        status)
            show_status
            ;;
        logs)
            if [[ -z "$2" ]]; then
                log_error "Especifica un servicio: $0 logs [servicio]"
                echo "Servicios: aegis-node, aegis-api, aegis-dashboard, aegis-admin"
                exit 1
            fi
            show_logs "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Comando desconocido: $1"
            show_help
            exit 1
            ;;
    esac
}

# Verificar que estamos en el directorio correcto
if [[ ! -f "production_config_v3.json" ]]; then
    log_error "Este script debe ejecutarse desde el directorio raíz de AEGIS"
    log_info "Asegúrate de estar en el directorio que contiene production_config_v3.json"
    exit 1
fi

# Ejecutar función principal
main "$@"