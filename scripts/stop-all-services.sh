#!/bin/bash

# ============================================================================
# AEGIS Framework - Script de Detención de Servicios (Linux)
# ============================================================================
# Descripción: Detiene todos los servicios del sistema AEGIS de forma segura
# Autor: AEGIS Security Team
# Versión: 1.0.0
# Fecha: 2024
# ============================================================================

set -euo pipefail

# Configuración de colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Variables globales
FORCE=false
VERBOSE=false
PIDS_FILE="./logs/service_pids.txt"
LOG_FILE="./logs/stop_services.log"

# Funciones de utilidad
log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[$timestamp] [INFO] $1" >> "$LOG_FILE" 2>/dev/null || true
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[$timestamp] [SUCCESS] $1" >> "$LOG_FILE" 2>/dev/null || true
}

log_warning() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[$timestamp] [WARNING] $1" >> "$LOG_FILE" 2>/dev/null || true
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$timestamp] [ERROR] $1" >> "$LOG_FILE" 2>/dev/null || true
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo -e "${CYAN}[DEBUG]${NC} $1"
        echo "[$timestamp] [DEBUG] $1" >> "$LOG_FILE" 2>/dev/null || true
    fi
}

show_help() {
    echo -e "${BLUE}=== AEGIS Framework - Detención de Servicios ===${NC}"
    echo ""
    echo -e "${YELLOW}USO:${NC}"
    echo "  $0 [OPCIONES]"
    echo ""
    echo -e "${YELLOW}OPCIONES:${NC}"
    echo "  --force     Forzar detención inmediata (SIGKILL)"
    echo "  --verbose   Mostrar salida detallada"
    echo "  --help      Mostrar esta ayuda"
    echo ""
    echo -e "${YELLOW}EJEMPLOS:${NC}"
    echo "  $0              # Detención normal"
    echo "  $0 --force      # Detención forzada"
    echo "  $0 --verbose    # Modo detallado"
    exit 0
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                ;;
            *)
                log_error "Opción desconocida: $1"
                show_help
                ;;
        esac
    done
}

stop_process_safely() {
    local service_name="$1"
    local pid="$2"
    local force_kill="$3"
    
    # Verificar si el proceso existe
    if ! kill -0 "$pid" 2>/dev/null; then
        log_warning "Proceso $service_name (PID: $pid) ya no existe"
        return 0
    fi
    
    log_info "Deteniendo $service_name (PID: $pid)..."
    
    if [[ "$force_kill" == "true" ]]; then
        log_debug "Forzando detención de $service_name"
        if kill -KILL "$pid" 2>/dev/null; then
            sleep 1
            if ! kill -0 "$pid" 2>/dev/null; then
                log_success "✅ $service_name detenido correctamente (forzado)"
                return 0
            fi
        fi
    else
        # Intentar detención suave primero
        log_debug "Intentando detención suave de $service_name"
        
        if kill -TERM "$pid" 2>/dev/null; then
            # Esperar hasta 10 segundos para detención suave
            local count=0
            while [[ $count -lt 10 ]] && kill -0 "$pid" 2>/dev/null; do
                sleep 1
                ((count++))
            done
            
            # Verificar si se detuvo
            if ! kill -0 "$pid" 2>/dev/null; then
                log_success "✅ $service_name detenido correctamente"
                return 0
            else
                log_warning "Detención suave falló, forzando detención de $service_name"
                if kill -KILL "$pid" 2>/dev/null; then
                    sleep 1
                    if ! kill -0 "$pid" 2>/dev/null; then
                        log_success "✅ $service_name detenido correctamente (forzado)"
                        return 0
                    fi
                fi
            fi
        fi
    fi
    
    log_error "❌ No se pudo detener $service_name (PID: $pid)"
    return 1
}

stop_process_by_name() {
    local process_name="$1"
    local command_filter="$2"
    local description="$3"
    
    log_info "Buscando $description..."
    
    local pids=()
    
    if [[ -n "$command_filter" ]]; then
        # Buscar por nombre y filtro de comando
        while IFS= read -r pid; do
            if [[ -n "$pid" ]]; then
                pids+=("$pid")
            fi
        done < <(pgrep -f "$process_name.*$command_filter" 2>/dev/null || true)
    else
        # Buscar solo por nombre
        while IFS= read -r pid; do
            if [[ -n "$pid" ]]; then
                pids+=("$pid")
            fi
        done < <(pgrep "$process_name" 2>/dev/null || true)
    fi
    
    if [[ ${#pids[@]} -eq 0 ]]; then
        log_info "No se encontraron procesos para $description"
        return
    fi
    
    log_info "Encontrados ${#pids[@]} procesos para $description"
    
    for pid in "${pids[@]}"; do
        stop_process_safely "$description" "$pid" "$FORCE"
    done
}

stop_services_from_pid_file() {
    if [[ ! -f "$PIDS_FILE" ]]; then
        log_warning "⚠️  Archivo de PIDs no encontrado: $PIDS_FILE"
        return
    fi
    
    log_info "📋 Leyendo servicios desde archivo de PIDs..."
    
    local stopped_count=0
    local total_count=0
    
    while IFS=':' read -r service_name pid; do
        if [[ -n "$service_name" && -n "$pid" ]]; then
            ((total_count++))
            log_debug "Procesando entrada: $service_name (PID: $pid)"
            
            if stop_process_safely "$service_name" "$pid" "$FORCE"; then
                ((stopped_count++))
            fi
        fi
    done < "$PIDS_FILE"
    
    log_success "✅ Detenidos $stopped_count de $total_count servicios del archivo PID"
    
    # Eliminar archivo de PIDs
    rm -f "$PIDS_FILE"
    log_debug "Archivo de PIDs eliminado"
}

stop_services_by_name() {
    log_info "🔍 Buscando servicios AEGIS por nombre de proceso..."
    
    # Definir servicios conocidos
    local services=(
        "python:main.py:Dashboard AEGIS"
        "node:vite:Secure Chat UI (Vite)"
        "node:hardhat:Blockchain Local (Hardhat)"
        "tor:torrc:Tor Service"
    )
    
    for service_def in "${services[@]}"; do
        IFS=':' read -r process_name command_filter description <<< "$service_def"
        stop_process_by_name "$process_name" "$command_filter" "$description"
    done
}

stop_additional_processes() {
    log_info "🧹 Limpiando procesos adicionales..."
    
    # Procesos que podrían quedar ejecutándose
    local additional_processes=("npm" "npx")
    
    for process_name in "${additional_processes[@]}"; do
        local pids=($(pgrep "$process_name" 2>/dev/null || true))
        
        if [[ ${#pids[@]} -gt 0 ]]; then
            log_info "Deteniendo procesos $process_name..."
            for pid in "${pids[@]}"; do
                stop_process_safely "$process_name" "$pid" "true"
            done
        fi
    done
}

check_remaining_processes() {
    log_info "🔍 Verificando procesos restantes..."
    
    local aegis_processes=()
    
    # Buscar procesos Python con main.py
    while IFS= read -r pid; do
        if [[ -n "$pid" ]]; then
            aegis_processes+=("Python AEGIS:$pid")
        fi
    done < <(pgrep -f "python.*main.py" 2>/dev/null || true)
    
    # Buscar procesos Node.js relacionados
    while IFS= read -r pid; do
        if [[ -n "$pid" ]]; then
            aegis_processes+=("Node.js AEGIS:$pid")
        fi
    done < <(pgrep -f "node.*(vite|hardhat)" 2>/dev/null || true)
    
    # Buscar procesos Tor
    while IFS= read -r pid; do
        if [[ -n "$pid" ]]; then
            aegis_processes+=("Tor:$pid")
        fi
    done < <(pgrep -f "tor.*torrc" 2>/dev/null || true)
    
    if [[ ${#aegis_processes[@]} -gt 0 ]]; then
        log_warning "⚠️  Procesos AEGIS aún ejecutándose:"
        for proc in "${aegis_processes[@]}"; do
            IFS=':' read -r name pid <<< "$proc"
            echo "   - $name (PID: $pid)"
        done
        
        if [[ "$FORCE" == "true" ]]; then
            log_info "Forzando detención de procesos restantes..."
            for proc in "${aegis_processes[@]}"; do
                IFS=':' read -r name pid <<< "$proc"
                stop_process_safely "$name" "$pid" "true"
            done
        else
            log_warning "💡 Usa --force para detener procesos restantes"
        fi
    else
        log_success "✅ No se encontraron procesos AEGIS ejecutándose"
    fi
}

test_ports_availability() {
    log_info "🔌 Verificando disponibilidad de puertos..."
    
    local ports=(8080 5173 8545 9050 9051)
    local busy_ports=()
    
    for port in "${ports[@]}"; do
        if nc -z localhost "$port" 2>/dev/null; then
            busy_ports+=("$port")
        fi
    done
    
    if [[ ${#busy_ports[@]} -gt 0 ]]; then
        log_warning "⚠️  Puertos aún ocupados: ${busy_ports[*]}"
        log_warning "   Algunos servicios podrían seguir ejecutándose"
        
        # Mostrar qué proceso está usando cada puerto
        for port in "${busy_ports[@]}"; do
            local process_info=$(lsof -ti:$port 2>/dev/null || true)
            if [[ -n "$process_info" ]]; then
                log_debug "Puerto $port usado por PID: $process_info"
            fi
        done
    else
        log_success "✅ Todos los puertos AEGIS están disponibles"
    fi
}

clean_log_files() {
    log_info "🧹 Limpiando archivos de log antiguos..."
    
    local log_files=(
        "./logs/dashboard.log"
        "./logs/secure-chat.log"
        "./logs/blockchain.log"
        "./logs/tor.log"
    )
    
    for log_file in "${log_files[@]}"; do
        if [[ -f "$log_file" ]]; then
            # Mantener solo las últimas 1000 líneas
            if tail -n 1000 "$log_file" > "${log_file}.tmp" 2>/dev/null; then
                mv "${log_file}.tmp" "$log_file"
                log_debug "Log truncado: $log_file"
            else
                rm -f "${log_file}.tmp" 2>/dev/null || true
                log_debug "No se pudo truncar log: $log_file"
            fi
        fi
    done
}

show_stop_summary() {
    echo ""
    echo -e "${BLUE}📊 Resumen de detención de servicios AEGIS:${NC}"
    echo -e "${BLUE}===========================================${NC}"
    
    log_success "🛑 Proceso de detención completado"
    
    # Mostrar estado de puertos
    test_ports_availability
    
    echo ""
    log_success "💡 Para reiniciar servicios:"
    echo "   ./scripts/start-all-services.sh"
    
    echo ""
    log_success "📋 Para verificar estado:"
    echo "   ps aux | grep -E '(python.*main.py|node.*(vite|hardhat)|tor.*torrc)'"
    
    echo ""
    log_success "📁 Logs disponibles en:"
    echo "   ./logs/"
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

main() {
    echo -e "${BLUE}🛑 AEGIS Framework - Deteniendo Servicios${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
    
    # Crear directorio de logs si no existe
    mkdir -p logs
    
    # Inicializar log
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] Iniciando detención de servicios AEGIS" > "$LOG_FILE"
    
    if [[ "$FORCE" == "true" ]]; then
        log_warning "⚠️  Modo FORZADO activado - detención inmediata"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "🔍 Modo VERBOSE activado"
    fi
    
    echo ""
    
    # 1. Detener servicios desde archivo PID (método preferido)
    stop_services_from_pid_file
    
    echo ""
    
    # 2. Buscar y detener servicios por nombre
    stop_services_by_name
    
    echo ""
    
    # 3. Limpiar procesos adicionales
    stop_additional_processes
    
    echo ""
    
    # 4. Verificar procesos restantes
    check_remaining_processes
    
    echo ""
    
    # 5. Limpiar logs si es necesario
    if [[ "$VERBOSE" != "true" ]]; then
        clean_log_files
    fi
    
    # 6. Mostrar resumen final
    show_stop_summary
    
    log_success "✅ ¡Detención de servicios completada!"
}

# Parsear argumentos y ejecutar
parse_arguments "$@"

# Verificar si se ejecuta como root (opcional warning)
if [[ $EUID -eq 0 ]]; then
    log_warning "⚠️  Ejecutándose como root - ten cuidado"
fi

# Ejecutar función principal
main