#!/bin/bash

# ============================================================================
# AEGIS Framework - Script de Inicio de Servicios (Linux)
# ============================================================================
# Descripción: Inicia todos los servicios del sistema AEGIS de forma ordenada
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
SKIP_TOR=false
SKIP_BLOCKCHAIN=false
SKIP_UI=false
VERBOSE=false
PIDS_FILE="./logs/service_pids.txt"

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

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

show_help() {
    echo -e "${BLUE}=== AEGIS Framework - Inicio de Servicios ===${NC}"
    echo ""
    echo -e "${YELLOW}USO:${NC}"
    echo "  $0 [OPCIONES]"
    echo ""
    echo -e "${YELLOW}OPCIONES:${NC}"
    echo "  --skip-tor          Omitir inicio de Tor"
    echo "  --skip-blockchain   Omitir inicio de blockchain local"
    echo "  --skip-ui           Omitir inicio de interfaces de usuario"
    echo "  --verbose           Mostrar salida detallada"
    echo "  --help              Mostrar esta ayuda"
    echo ""
    echo -e "${YELLOW}EJEMPLOS:${NC}"
    echo "  $0                          # Iniciar todos los servicios"
    echo "  $0 --skip-tor              # Iniciar sin Tor"
    echo "  $0 --verbose               # Modo detallado"
    echo "  $0 --skip-ui --skip-tor    # Solo dashboard y blockchain"
    exit 0
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tor)
                SKIP_TOR=true
                shift
                ;;
            --skip-blockchain)
                SKIP_BLOCKCHAIN=true
                shift
                ;;
            --skip-ui)
                SKIP_UI=true
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

check_prerequisites() {
    log_info "🔍 Verificando prerequisitos..."
    
    local errors=()
    
    # Verificar Python
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version 2>&1)
        log_success "✅ Python: $python_version"
    else
        errors+=("Python3 no encontrado")
    fi
    
    # Verificar Node.js
    if command -v node &> /dev/null; then
        local node_version=$(node --version 2>&1)
        log_success "✅ Node.js: $node_version"
    else
        errors+=("Node.js no encontrado")
    fi
    
    # Verificar npm
    if command -v npm &> /dev/null; then
        local npm_version=$(npm --version 2>&1)
        log_success "✅ npm: $npm_version"
    else
        errors+=("npm no encontrado")
    fi
    
    # Verificar entorno virtual de Python
    if [[ -f "venv/bin/activate" ]]; then
        log_success "✅ Entorno virtual Python encontrado"
    else
        errors+=("Entorno virtual Python no encontrado (venv/)")
    fi
    
    # Verificar archivos de configuración
    local config_files=(".env" "config/app_config.json")
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "✅ Configuración: $file"
        else
            errors+=("Archivo de configuración faltante: $file")
        fi
    done
    
    # Verificar Tor si no se omite
    if [[ "$SKIP_TOR" == "false" ]]; then
        if command -v tor &> /dev/null; then
            local tor_version=$(tor --version 2>&1 | head -n1)
            log_success "✅ Tor: $tor_version"
        else
            log_warning "⚠️  Tor no encontrado (se omitirá automáticamente)"
            SKIP_TOR=true
        fi
    fi
    
    if [[ ${#errors[@]} -gt 0 ]]; then
        log_error "❌ Errores encontrados:"
        for error in "${errors[@]}"; do
            echo "   - $error"
        done
        echo ""
        log_warning "💡 Ejecuta el script de despliegue primero:"
        echo "   ./scripts/auto-deploy-linux.sh"
        exit 1
    fi
    
    log_success "✅ Todos los prerequisitos verificados"
    echo ""
}

create_directories() {
    local dirs=("logs" "tor_data")
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_success "📁 Directorio $dir creado"
        fi
    done
    
    # Limpiar archivo de PIDs anterior
    > "$PIDS_FILE"
}

start_tor_service() {
    if [[ "$SKIP_TOR" == "true" ]]; then
        log_warning "⏭️  Omitiendo inicio de Tor"
        return
    fi
    
    log_info "🧅 Iniciando servicio Tor..."
    
    # Verificar si Tor ya está ejecutándose
    if pgrep -f "tor.*torrc" > /dev/null; then
        local tor_pid=$(pgrep -f "tor.*torrc")
        log_warning "⚠️  Tor ya está ejecutándose (PID: $tor_pid)"
        return
    fi
    
    # Verificar configuración de Tor
    if [[ ! -f "config/torrc" ]]; then
        log_error "❌ Archivo config/torrc no encontrado"
        return
    fi
    
    # Configurar permisos del directorio de datos
    chmod 700 tor_data 2>/dev/null || true
    
    # Iniciar Tor en segundo plano
    log_debug "Ejecutando: tor -f config/torrc"
    
    if [[ "$VERBOSE" == "true" ]]; then
        tor -f config/torrc > logs/tor.log 2>&1 &
    else
        tor -f config/torrc > logs/tor.log 2>&1 &
    fi
    
    local tor_pid=$!
    echo "tor:$tor_pid" >> "$PIDS_FILE"
    
    # Esperar un momento para verificar que inició correctamente
    sleep 3
    
    if kill -0 "$tor_pid" 2>/dev/null; then
        log_success "✅ Tor iniciado correctamente (PID: $tor_pid)"
        echo "   SOCKS Proxy: 127.0.0.1:9050"
        echo "   Control Port: 127.0.0.1:9051"
    else
        log_error "❌ Error al iniciar Tor"
        if [[ "$VERBOSE" == "true" ]]; then
            tail -n 10 logs/tor.log
        fi
    fi
    
    echo ""
}

start_dashboard() {
    log_info "🖥️  Iniciando Dashboard AEGIS..."
    
    # Activar entorno virtual y iniciar dashboard
    log_debug "Activando entorno virtual y ejecutando dashboard"
    
    if [[ "$VERBOSE" == "true" ]]; then
        source venv/bin/activate && python main.py start-dashboard --config config/app_config.json > logs/dashboard.log 2>&1 &
    else
        source venv/bin/activate && python main.py start-dashboard --config config/app_config.json > logs/dashboard.log 2>&1 &
    fi
    
    local dashboard_pid=$!
    echo "dashboard:$dashboard_pid" >> "$PIDS_FILE"
    
    # Esperar un momento para verificar que inició
    sleep 5
    
    if kill -0 "$dashboard_pid" 2>/dev/null; then
        log_success "✅ Dashboard iniciado correctamente (PID: $dashboard_pid)"
        echo "   URL: http://localhost:8080"
    else
        log_error "❌ Error al iniciar Dashboard"
        if [[ "$VERBOSE" == "true" ]]; then
            tail -n 10 logs/dashboard.log
        fi
    fi
    
    echo ""
}

start_secure_chat_ui() {
    if [[ "$SKIP_UI" == "true" ]]; then
        log_warning "⏭️  Omitiendo inicio de Secure Chat UI"
        return
    fi
    
    log_info "💬 Iniciando Secure Chat UI..."
    
    # Verificar directorio
    if [[ ! -d "dapps/secure-chat/ui" ]]; then
        log_error "❌ Directorio dapps/secure-chat/ui no encontrado"
        return
    fi
    
    # Verificar node_modules
    if [[ ! -d "dapps/secure-chat/ui/node_modules" ]]; then
        log_warning "⚠️  Dependencias no instaladas, instalando..."
        cd dapps/secure-chat/ui
        npm install
        cd ../../..
    fi
    
    # Cambiar al directorio y iniciar
    cd dapps/secure-chat/ui
    
    log_debug "Ejecutando: npm run dev"
    
    if [[ "$VERBOSE" == "true" ]]; then
        npm run dev > ../../../logs/secure-chat.log 2>&1 &
    else
        npm run dev > ../../../logs/secure-chat.log 2>&1 &
    fi
    
    local ui_pid=$!
    cd ../../..
    echo "secure-chat:$ui_pid" >> "$PIDS_FILE"
    
    # Esperar un momento para verificar que inició
    sleep 8
    
    if kill -0 "$ui_pid" 2>/dev/null; then
        log_success "✅ Secure Chat UI iniciado correctamente (PID: $ui_pid)"
        echo "   URL: http://localhost:5173"
    else
        log_error "❌ Error al iniciar Secure Chat UI"
        if [[ "$VERBOSE" == "true" ]]; then
            tail -n 10 logs/secure-chat.log
        fi
    fi
    
    echo ""
}

start_blockchain_node() {
    if [[ "$SKIP_BLOCKCHAIN" == "true" ]]; then
        log_warning "⏭️  Omitiendo inicio de blockchain local"
        return
    fi
    
    log_info "⛓️  Iniciando nodo blockchain local..."
    
    # Verificar directorio
    if [[ ! -d "dapps/aegis-token" ]]; then
        log_error "❌ Directorio dapps/aegis-token no encontrado"
        return
    fi
    
    # Verificar node_modules
    if [[ ! -d "dapps/aegis-token/node_modules" ]]; then
        log_warning "⚠️  Dependencias no instaladas, instalando..."
        cd dapps/aegis-token
        npm install
        cd ../..
    fi
    
    # Cambiar al directorio y iniciar
    cd dapps/aegis-token
    
    log_debug "Ejecutando: npx hardhat node"
    
    if [[ "$VERBOSE" == "true" ]]; then
        npx hardhat node > ../../logs/blockchain.log 2>&1 &
    else
        npx hardhat node > ../../logs/blockchain.log 2>&1 &
    fi
    
    local blockchain_pid=$!
    cd ../..
    echo "blockchain:$blockchain_pid" >> "$PIDS_FILE"
    
    # Esperar un momento para verificar que inició
    sleep 10
    
    if kill -0 "$blockchain_pid" 2>/dev/null; then
        log_success "✅ Nodo blockchain iniciado correctamente (PID: $blockchain_pid)"
        echo "   RPC URL: http://localhost:8545"
        echo "   Chain ID: 31337"
    else
        log_error "❌ Error al iniciar nodo blockchain"
        if [[ "$VERBOSE" == "true" ]]; then
            tail -n 10 logs/blockchain.log
        fi
    fi
    
    echo ""
}

test_service_health() {
    log_info "🏥 Verificando salud de servicios..."
    
    # Función para probar HTTP
    test_http_service() {
        local name="$1"
        local url="$2"
        local skip="$3"
        
        if [[ "$skip" == "true" ]]; then
            echo -e "⏭️  $name: Omitido"
            return
        fi
        
        if curl -s --max-time 5 --head "$url" > /dev/null 2>&1; then
            log_success "✅ $name: Funcionando"
        else
            log_error "❌ $name: No responde"
        fi
    }
    
    # Probar servicios HTTP
    test_http_service "Dashboard" "http://localhost:8080" "false"
    test_http_service "Secure Chat UI" "http://localhost:5173" "$SKIP_UI"
    test_http_service "Blockchain RPC" "http://localhost:8545" "$SKIP_BLOCKCHAIN"
    
    # Verificar Tor SOCKS proxy si no se omitió
    if [[ "$SKIP_TOR" == "false" ]]; then
        if nc -z 127.0.0.1 9050 2>/dev/null; then
            log_success "✅ Tor SOCKS Proxy: Funcionando"
        else
            log_error "❌ Tor SOCKS Proxy: No responde"
        fi
    fi
    
    echo ""
}

show_service_status() {
    echo -e "${BLUE}📊 Estado de servicios AEGIS:${NC}"
    echo -e "${BLUE}================================${NC}"
    
    # Mostrar procesos activos
    if [[ -f "$PIDS_FILE" ]]; then
        log_success "🔄 Servicios en ejecución:"
        while IFS=':' read -r service pid; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "   $service (PID: $pid)"
            else
                echo "   $service (PID: $pid) - ❌ No ejecutándose"
            fi
        done < "$PIDS_FILE"
    else
        log_warning "⚠️  No hay información de servicios"
    fi
    
    echo ""
    
    # URLs de acceso
    log_success "🌐 URLs de acceso:"
    echo "   Dashboard:      http://localhost:8080"
    if [[ "$SKIP_UI" == "false" ]]; then
        echo "   Secure Chat:    http://localhost:5173"
    fi
    if [[ "$SKIP_BLOCKCHAIN" == "false" ]]; then
        echo "   Blockchain RPC: http://localhost:8545"
    fi
    if [[ "$SKIP_TOR" == "false" ]]; then
        echo "   Tor SOCKS:      127.0.0.1:9050"
        echo "   Tor Control:    127.0.0.1:9051"
    fi
    
    echo ""
    log_warning "💡 Para detener servicios:"
    echo "   ./scripts/stop-all-services.sh"
    echo "   O presiona Ctrl+C"
    echo ""
    log_warning "📋 Para ver logs:"
    echo "   tail -f logs/dashboard.log"
    echo "   tail -f logs/error.log"
}

cleanup_on_exit() {
    echo ""
    log_warning "🛑 Deteniendo servicios..."
    
    if [[ -f "$PIDS_FILE" ]]; then
        while IFS=':' read -r service pid; do
            if kill -0 "$pid" 2>/dev/null; then
                log_info "Deteniendo $service (PID: $pid)"
                kill -TERM "$pid" 2>/dev/null || true
                
                # Esperar un momento y forzar si es necesario
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
        done < "$PIDS_FILE"
        
        rm -f "$PIDS_FILE"
    fi
    
    log_success "✅ Servicios detenidos"
    exit 0
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

main() {
    # Configurar trap para limpieza
    trap cleanup_on_exit SIGINT SIGTERM EXIT
    
    echo -e "${BLUE}🚀 AEGIS Framework - Iniciando Servicios${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
    
    # Verificar prerequisitos
    check_prerequisites
    
    # Crear directorios necesarios
    create_directories
    
    # Iniciar servicios en orden
    log_info "🔄 Iniciando servicios..."
    echo ""
    
    # 1. Tor (si está habilitado)
    start_tor_service
    
    # 2. Dashboard (siempre)
    start_dashboard
    
    # 3. Blockchain (si está habilitado)
    start_blockchain_node
    
    # 4. UI (si está habilitado)
    start_secure_chat_ui
    
    # Esperar un momento para que todos los servicios se estabilicen
    log_warning "⏳ Esperando estabilización de servicios..."
    sleep 10
    
    # Verificar salud de servicios
    test_service_health
    
    # Mostrar estado final
    show_service_status
    
    log_success "🎉 ¡Todos los servicios iniciados!"
    echo ""
    log_warning "⚠️  IMPORTANTE: Mantén esta terminal abierta para que los servicios sigan funcionando"
    log_warning "   Presiona Ctrl+C para detener todos los servicios"
    
    # Mantener el script ejecutándose
    echo ""
    echo "Presiona Ctrl+C para detener todos los servicios..."
    
    while true; do
        sleep 30
        
        # Verificar que los procesos sigan ejecutándose
        local running_count=0
        if [[ -f "$PIDS_FILE" ]]; then
            while IFS=':' read -r service pid; do
                if kill -0 "$pid" 2>/dev/null; then
                    ((running_count++))
                fi
            done < "$PIDS_FILE"
        fi
        
        if [[ $running_count -eq 0 ]]; then
            log_warning "⚠️  Todos los servicios han terminado"
            break
        fi
    done
}

# Parsear argumentos y ejecutar
parse_arguments "$@"
main